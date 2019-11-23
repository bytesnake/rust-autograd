use crate::ndarray_ext::{NdArray, NdArrayView, NdArrayViewMut};
use crate::op;
use crate::tensor::Tensor;
use crate::Float;
use crate::{hashbrown::hash_map::Entry, FxHashMap, FxHashSet};
use crossbeam::crossbeam_channel;
use ndarray;
use std::cell::Cell;
use std::cell::UnsafeCell;
use std::mem;
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

/// Helper structure for batched evaluation.
///
/// Use this in case `ag::eval` doesn't help.
///
/// ```
/// use autograd as ag;
/// use ndarray;
///
/// let ref a = ag::placeholder(&[]);
/// let ref x = a + a;
/// let ref y = a * a;
/// let ref z = a / a;
///
/// ag::Eval::new()
///     .push(&y)
///     .extend(&[y, z])
///     .run(&[ag::Feed(a, ndarray::arr0(2.).into_dyn().view())]);  // Do eval
/// ```
pub struct Eval<'k, T: Float> {
    buf: Vec<&'k Tensor<T>>,
}

impl<'c, 'k, 'v, T: Float> Eval<'k, T> {
    #[inline]
    /// Instantiates a new evaluation session.
    pub fn new() -> Self {
        Eval { buf: Vec::new() }
    }

    #[inline]
    /// Appends a tensor to the back of the evaluation targets.
    pub fn push(&mut self, x: &'k Tensor<T>) -> &mut Self {
        self.buf.push(x);
        self
    }

    #[inline]
    /// Extends the evaluation targets with `xs`.
    pub fn extend<A>(&mut self, xs: &'k [A]) -> &mut Self
        where
            A: AsRef<Tensor<T>>,
    {
        self.buf.extend(xs.iter().map(|x| x.as_ref()));
        self
    }

    #[inline]
    /// Evaluates the buffered tensors.
    ///
    /// `feeds` is a stream of `(placeholder tensor, its value)`
    pub fn run(&'k self, feeds: &'c [crate::runtime::Feed<'k, 'v, T>]) -> Vec<Option<NdArray<T>>> {
        eval(&self.buf, feeds)
    }
}

pub enum OpInput<'v, T: Float> {
    RO(Option<NdArrayView<'v, T>>),    // Read-only view
    RW(Option<NdArrayViewMut<'v, T>>), // Read-write view
}

impl<'v, T: Float> OpInput<'v, T> {
    #[inline]
    pub fn new(x: NdArrayView<'v, T>) -> Self {
        OpInput::RO(Some(x))
    }

    #[inline]
    pub fn new_mut(x: NdArrayViewMut<'v, T>) -> Self {
        OpInput::RW(Some(x))
    }
}

pub struct OpComputeContext<'k, 'v, T: Float> {
    pub(crate) target: &'k Tensor<T>,
    pub(crate) xs: Vec<OpInput<'v, T>>,
    pub(crate) ys: Option<Vec<Result<crate::ArrRepr<'v, T>, crate::op::ComputeException>>>,
}

impl<'k, 'v, T: Float> OpComputeContext<'k, 'v, T> {

    #[inline]
    pub fn input(&mut self, i: usize) -> NdArrayView<'v, T> {
        let x = match self.xs.get_mut(i) {
            Some(x) => x,
            None => panic!(
                "Bad op impl of {}: input index out of range.",
                self.target.op.name()
            ),
        };
        match x {
            OpInput::RO(ref mut a) => match mem::replace(a, None) {
                Some(ret) => ret,
                None => panic!(
                    "Bad op impl of {}: OpComputeContext::input({}) cannot be called twice",
                    self.target.op.name(),
                    i
                ),
            },
            _ => {
                panic!(
                    "Bad op impl: cannot perform immutable borrowing for input({})",
                    i
                );
            }
        }
    }

    #[inline]
    pub fn input_mut(&mut self, i: usize) -> NdArrayViewMut<'v, T> {
        let x = match self.xs.get_mut(i) {
            Some(x) => x,
            None => panic!(
                "Bad op impl of {}: {}'s input doesn't exist.",
                self.target.op.name(),
                i
            ),
        };
        match x {
            OpInput::RW(ref mut a) => match mem::replace(a, None) {
                Some(ret) => ret,
                None => panic!(
                    "Bad op impl of {}: OpComputeContext::input_mut({}) cannot be called twice",
                    self.target.op.name(),
                    i
                ),
            },
            _ => {
                panic!(
                    "Bad op impl of {}: cannot perform mutable borrowing for input({})",
                    self.target.op.name(),
                    i
                );
            }
        }
    }

    #[inline]
    pub(crate) fn new(target: &'k Tensor<T>, xs: Vec<OpInput<'v, T>>) -> Self {
        OpComputeContext {
            target,
            xs,
            ys: None,
        }
    }

    #[inline]
    pub fn set_output(
        &mut self,
        ys: Vec<Result<crate::ArrRepr<'v, T>, crate::op::ComputeException>>,
    ) {
        self.ys = Some(ys)
    }

    #[inline]
    pub fn num_inputs(&self) -> usize {
        self.xs.len()
    }

    #[inline]
    fn node(&self) -> &Tensor<T> {
        self.target
    }
}

#[derive(Debug)]
struct Node<'a, T: Float> {
    inner: &'a Tensor<T>,
    successors: FxHashSet<&'a Tensor<T>>,
    // initialized with the number of the immediate predecessors.
    // When this is reduced to 0, `node` is ready to be evaluated.
    pending_count: Cell<usize>,
    scheduled: Cell<bool>,
}

use std::ops::{Deref, DerefMut};

impl<'a, T: Float> Deref for Node<'a, T> {
    type Target = Tensor<T>;
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        self.inner
    }
}

impl<'a, T: Float> Node<'a, T> {
    #[inline]
    fn new(inner: &'a Tensor<T>, successor: Option<&'a Tensor<T>>) -> Self {
        let mut successors = FxHashSet::default();
        if let Some(successor) = successor {
            successors.insert(successor);
        }
        Node {
            inner,
            successors,
            pending_count: Cell::new(0),
            scheduled: Cell::new(false),
        }
    }

    #[inline]
    fn scheduled(&self) -> bool {
        self.scheduled.get()
    }

    #[inline]
    fn mark_scheduled(&self) {
        self.scheduled.set(true);
    }

    #[inline]
    fn increment_pending_count(&self) {
        self.pending_count.set(self.pending_count.get() + 1);
    }

    #[inline]
    fn decrement_pending_count(&self) {
        self.pending_count
            .set(self.pending_count.get().saturating_sub(1));
    }

    #[inline]
    fn ready(&self) -> bool {
        self.pending_count.get() == 0
    }
}

// Builds a subgraph consisting of nodes that are reachable from `tensors`.
// Returns: graph, source_nodes
fn build_minimum_subgraph_from<T, F>(targets: &[T]) -> Graph<F>
    where
        T: AsRef<Tensor<F>>,
        F: Float,
{
    let mut edge_info = FxHashMap::<&Tensor<F>, Node<F>>::default();
    let mut sources = FxHashSet::default();
    let mut dfs_stack: Vec<&Tensor<_>> = Vec::with_capacity(128);

    // Initialize the graph and stack with `targets`
    for t in targets {
        let t = t.as_ref();
        let node = Node::new(t, None);
        if let Entry::Vacant(ent) = edge_info.entry(t) {
            let inserted = ent.insert(node);
            dfs_stack.push(inserted.inner);
        } else {
            panic!("Detected a duplication in the given evaluation target list.");
        }
    }

    while let Some(node) = dfs_stack.pop() {
        if node.is_source() {
            sources.insert(node);
        }
        for child in &node.inputs {
            let mut found_new_successor = true;
            match edge_info.entry(&child.val) {
                Entry::Vacant(ent) => {
                    // initial visit
                    let inserted = ent.insert(Node::new(&child.val, Some(node)));
                    dfs_stack.push(inserted.inner);
                }
                Entry::Occupied(mut ent) => {
                    let successors = &mut ent.get_mut().successors;
                    // ensuring no duplication in successors to handle the case like `y = add(x, x)`.
                    // linear search is enough here.
                    if !successors.contains(&node) {
                        successors.insert(node); // Make a edge
                    } else {
                        found_new_successor = false;
                    }
                }
            }
            if found_new_successor {
                edge_info.get_mut(&node).unwrap().increment_pending_count();
            }
        }
    }
    Graph {
        edge_info,
        sources
    }
}

#[inline]
fn contains<T: PartialEq>(slice: &[T], item: T) -> bool {
    for x in slice {
        if *x == item {
            return true;
        }
    }
    false
}

struct OpEvalResult<'k, 'v, T: Float> {
    node: &'k Tensor<T>,
    ys: crate::op::ComputeResults<'v, T>,
}

struct LockState<'tensor, 'lock, F: Float> {
    read_guards: UnsafeCell<FxHashMap<&'tensor Tensor<F>, Vec<Option<RwLockReadGuard<'lock, NdArray<F>>>>>>,
    write_guards: UnsafeCell<FxHashMap<&'tensor Tensor<F>, Vec<Option<RwLockWriteGuard<'lock, NdArray<F>>>>>>,
}

impl<'tensor, 'lock, F: Float> LockState<'tensor, 'lock, F> {

    #[inline]
    fn insert_mut(&self,
                  key: &'tensor Tensor<F>,
                  input_idx: usize,
                  lock: &'lock RwLock<NdArray<F>>) -> &mut RwLockWriteGuard<'lock, NdArray<F>> {
        unsafe {
            let mut got_mut: &mut Vec<Option<_>> = (&mut *self.write_guards.get()).get_mut(key).unwrap();
            got_mut[input_idx] = Some(lock.write().unwrap());
            got_mut[input_idx].as_mut().unwrap()
        }
    }

    #[inline]
    fn insert(
        &self,
        key: &'tensor Tensor<F>,
        input_idx: usize,
        lock: &'lock RwLock<NdArray<F>>
    ) -> &RwLockReadGuard<'lock, NdArray<F>> {
        unsafe {
            let mut got_mut: &mut Vec<Option<_>> = (&mut *self.read_guards.get()).get_mut(key).unwrap();
            got_mut[input_idx] = Some(lock.read().unwrap());
            got_mut[input_idx].as_ref().unwrap()
        }
    }

    #[inline]
    fn init(&self, key: &'tensor Tensor<F>) {
        unsafe {
            (&mut *self.write_guards.get()).insert(key, crate::none_vec(key.inputs.len()));
            (&mut *self.read_guards.get()).insert(key, crate::none_vec(key.inputs.len()));
        }
    }

    #[inline]
    fn new() -> Self {
        LockState {
            read_guards: UnsafeCell::new(FxHashMap::default()),
            write_guards: UnsafeCell::new(FxHashMap::default()),
        }
    }

    #[inline]
    fn invalidate(&self, mut_usage: bool, node: &Tensor<F>, i: usize) {
        unsafe {
            if mut_usage {
                mem::swap(
                    &mut (&mut *self.write_guards.get()).get_mut(node).unwrap()[i],
                    &mut None,
                );
            } else {
                mem::swap(
                    &mut (&mut *self.read_guards.get()).get_mut(node).unwrap()[i],
                    &mut None,
                );
            }
        }
    }
}

#[inline]
fn is_eval_target<A, T>(node: &Tensor<T>, targets: &[A]) -> bool
    where
        A: AsRef<Tensor<T>>,
        T: Float,
{
    for t in targets {
        if node == t.as_ref() {
            // comparing node ids
            return true;
        }
    }
    false
}

struct Graph<'tensor, F: Float> {
    // source nodes in this graph
    sources: FxHashSet<&'tensor Tensor<F>>,
    // tensor -> Node(=its immediate successor nodes etc)
    edge_info: FxHashMap<&'tensor Tensor<F>, Node<'tensor, F>>
}

impl<'a, F: Float> Graph<'a, F> {
    #[inline]
    fn get(&self, key: &'a Tensor<F>) -> &Node<'a, F> {
        self.edge_info.get(key).unwrap()
    }
}

pub fn eval<'slice, 'node, 'feed, K, T>(
    tensors: &'node [K],
    feeds: &'slice [Feed<'node, 'feed, T>],
) -> Vec<Option<NdArray<T>>>
    where
        K: AsRef<Tensor<T>>,
        T: Float,
{
    let mut graph = build_minimum_subgraph_from(tensors);

    let mut output_info_storage = FxHashMap::<&Tensor<T>, Vec<ValueInfo>>::default();

    // Storage in which compute results are stored.
    let mut owned_storage = UnsafeCell::new(Vec::<Option<NdArray<T>>>::with_capacity(100));

    let mut lock_state = LockState::new();

    // Views of owned arrays whose lifetime is shorter than owned ones
    let mut view_storage = Vec::<NdArrayView<T>>::new();

    let (sender, receiver) = crossbeam_channel::bounded(16);

    for &src in &graph.sources {
        // source: placeholder, constant, variable, generator.
        sender.send(OpEvalResult {
            node: src,
            ys: if !src.requires_compute() {
                vec![]
            } else {
                let mut ctx = OpComputeContext::new(src, vec![]);
                src.op.compute(&mut ctx);
                ctx.ys.unwrap()
            },
        });
    }

    let mut targets_remaining = tensors.len();

    loop {
        // Aggregate and register a compute result.
        let evaluated: &Node<T> = unsafe {
            let OpEvalResult { node, ys } = receiver.recv().unwrap();
            for (i, input) in node.inputs.iter().enumerate() {
                lock_state.invalidate(input.mut_usage, node, i);
            }
            if node.requires_compute() {
                let mut info_list = Vec::with_capacity(ys.len());
                for y in ys {
                    let info = match y {
                        Ok(crate::ArrRepr::Owned(val)) => {
                            (&mut *owned_storage.get()).push(Some(val));
                            ValueInfo::new(ValueType::Owned, (&*owned_storage.get()).len() - 1)
                        }
                        Ok(crate::ArrRepr::View(val)) => {
                            view_storage.push(val);
                            ValueInfo::new(ValueType::View, view_storage.len() - 1)
                        }
                        _ => ValueInfo::new(ValueType::Empty, /*dummy = */ 0)
                    };
                    info_list.push(info);
                }
                output_info_storage.insert(node, info_list);
            }
            graph.get(node)
        };

        if is_eval_target(evaluated.inner, tensors) {
            targets_remaining -= 1;
            if targets_remaining == 0 {
                break; // exit the main loop as all the target tensors were evaluated.
            }
        }

        // Try to schedule the immediate successors of the evaluated node.
        for &suc in &evaluated.successors {
            let suc_info = graph.get(&suc);
            suc_info.decrement_pending_count();

            if !suc_info.scheduled() && suc_info.ready() {
                suc_info.mark_scheduled();
                lock_state.init(suc);

                let mut xs = Vec::with_capacity(suc.inputs.len());

                // Aggregate inputs for `in_node`
                for (i, (in_node, &in_idx)) in suc.inputs.iter().zip(&suc.input_indices).enumerate() {
                    if in_node.is_placeholder {
                        for feed in feeds {
                            // linear search is enough for feeds
                            if Arc::ptr_eq(feed.0, &in_node.val) {
                                let clone = feed.1.view();
                                if !in_node
                                    .known_shape
                                    .as_ref()
                                    .unwrap()
                                    .validate(clone.shape())
                                {
                                    panic!(
                                        "Shape error: placeholder required {:?}, but got {:?}",
                                        in_node.known_shape.as_ref().unwrap().get(),
                                        clone.shape()
                                    );
                                }
                                xs.push(OpInput::new(clone));
                                break;
                            }
                        }
                    } else if let Some(ref lock) = in_node.variable_array {
                        unsafe {
                            if in_node.mut_usage {
                                xs.push(OpInput::new_mut(lock_state.insert_mut(suc, i, lock).view_mut()));
                            } else {
                                xs.push(OpInput::new(lock_state.insert(suc, i, lock).view()));
                            }
                        }
                    } else if let Some(ref arr) = in_node.get_constant_array() {
                        xs.push(OpInput::new(arr.view()));
                    } else {
                        // Search the output of other nodes
                        let info = &output_info_storage.get(&&in_node.val).unwrap()[in_idx];
                        match info.ty {
                            ValueType::Owned => unsafe {
                                xs.push(OpInput::new((&*owned_storage.get())[info.key].as_ref().unwrap().view()));
                            },
                            ValueType::View => {
                                xs.push(OpInput::new(view_storage[info.key].clone()));
                            }
                            ValueType::Empty => {
                                panic!(
                                    "{}'s output, which is empty, was fed to {}",
                                    in_node.op.name(),
                                    suc.op.name()
                                );
                            }
                        }
                    }
                }

                crate::rayon::scope(|s| {
                    s.spawn(|_| {
                        // run compute
                        let mut ctx = OpComputeContext::new(suc, xs);
                        suc.op.compute(&mut ctx);
                        sender.send(OpEvalResult {
                            node: suc,
                            ys: ctx.ys.expect("Bad op implementation: empty return value"),
                        });
                    })
                });
            }
        }
    }

    let owned_storage = unsafe { &mut *owned_storage.get() };

    let mut ret: Vec<Option<NdArray<T>>> = Vec::with_capacity(tensors.len());
    for t in tensors {
        let t = t.as_ref();
        let arr = if let Some(per) = t.clone_persistent_array() {
            // rare case
            Some(per)
        } else if t.is_placeholder {
            // rare case
            let mut found = None;
            for feed in feeds {
                if Arc::ptr_eq(feed.0, t) {
                    found = Some(&feed.1);
                    break;
                }
            }
            Some(found.expect("Placeholder unfilled.").to_owned())
        } else {
            let info = &output_info_storage.get(&t).unwrap()[0];
            if ValueType::Owned == info.ty {
                mem::replace(&mut owned_storage[info.key], None)
            } else if ValueType::View == info.ty {
                Some(view_storage[info.key].to_owned())
            } else {
                None
            }
        };
        ret.push(arr);
    }
    ret
}

#[derive(Clone, Copy, PartialEq)]
enum ValueType {
    Owned,
    View,
    Empty,
}

#[derive(Clone)]
struct ValueInfo {
    ty: ValueType,
    // key to lookup the value
    key: usize,
}

impl ValueInfo {
    #[inline]
    fn new(ty: ValueType, key: usize) -> Self {
        Self {
            ty, key
        }
    }
}

/// An object sent to `ag::Eval`, `ag::eval` or `Tensor::eval` to fill a placeholder tensor
///
/// The first argument should be a tensor made from `ag::placeholder`.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let x = ag::placeholder(&[2]);
///
/// // Fills placeholder, then eval
/// let arr = ndarray::arr1(&[1., 1.]).into_dyn();
/// x.eval(&[ag::Feed(&x, arr.view())]);
/// ```
pub struct Feed<'k, 'f, T: Float>(
    pub &'k Tensor<T>,                             // a placeholder tensor
    pub ndarray::ArrayView<'f, T, ndarray::IxDyn>, // its value
);

#[test]
fn test_eval() {
    let ref v = crate::ops::placeholder::<f32>(&[3, 2, 1]);
    let ref z = crate::ops::reduce_sum(&crate::ops::squeeze(v, &[2]), &[0, 1], false);
    let ref g = crate::ops::grad(&[z], &[v]);
    let eval_result = &eval(g, &[Feed(v, crate::ndarray_ext::ones(&[3, 2, 1]).view())])[0];
    assert_eq!(eval_result.as_ref().unwrap().shape(), &[3, 2, 1]);
}

#[test]
fn test_constant_eval() {
    let arr = ndarray::arr1(&[0., 0., 0.]);
    assert_eq!(Some(arr.clone().into_dyn()), crate::variable(arr).eval(&[]));
}

#[test]
fn test_placeholder_eval() {
    let arr = crate::ndarray_ext::ones::<f32>(&[3, 2, 1]);
    let ref v = crate::ops::placeholder(&[3, 2, 1]);
    let eval_result = eval(&[v], &[Feed(v, arr.view())]);
    assert_eq!(eval_result[0], Some(arr));
}