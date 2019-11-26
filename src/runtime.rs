use crate::ndarray_ext::{NdArray, NdArrayView, NdArrayViewMut};
use crate::tensor::Tensor;
use crate::Float;
use crate::{hashbrown::hash_map::Entry, FxHashMap, FxHashSet};
use crate::arrayvec::ArrayVec;
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
///     .push(&x)
///     .extend(&[y, z])
///     .run(&[ag::Feed(a, ndarray::arr0(2.).into_dyn().view())]);  // Do eval
/// ```
pub struct Eval<'k, T: Float> {
    buf: Vec<&'k Tensor<'k, T>>,
}

impl<'c, 'k, 'v, T: Float> Eval<'k, T> {
    #[inline]
    /// Instantiates a new evaluation session.
    pub fn new() -> Self {
        Eval { buf: Vec::new() }
    }

    #[inline]
    /// Appends a tensor to the back of the evaluation targets.
    pub fn push(&mut self, x: &'k Tensor<'k, T>) -> &mut Self {
        self.buf.push(x);
        self
    }

    #[inline]
    /// Extends the evaluation targets with `xs`.
    pub fn extend<A>(&mut self, xs: &'k [&'k Tensor<'k, T>]) -> &mut Self
    where
        A: AsRef<Tensor<'k, T>>,
    {
        self.buf.extend(xs);
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
pub struct Feed<'t, 'feed, T: Float>(
    pub &'t Tensor<'t, T>, // a placeholder tensor
    pub ndarray::ArrayView<'feed, T, ndarray::IxDyn>, // its value
);

/// Wrapper object of NdArrayView/NdArrayViewMut which is fed to `Op::compute`
///
/// Used in `OpComputeContext`.
pub enum OpInput<'v, T: Float> {
    /// Read-only view
    RO(Option<NdArrayView<'v, T>>),
    /// Read-write view
    RW(Option<NdArrayViewMut<'v, T>>),
}

impl<'v, T: Float> OpInput<'v, T> {
    #[inline]
    /// Make a read-only input array
    pub fn new(x: NdArrayView<'v, T>) -> Self {
        OpInput::RO(Some(x))
    }

    #[inline]
    /// Make a read/write input array
    pub fn new_mut(x: NdArrayViewMut<'v, T>) -> Self {
        OpInput::RW(Some(x))
    }
}

/// Holds input/output arrays for an `Op`, and a `Tensor` representation of an `Op`.
pub struct OpComputeContext<'v, T: Float> {
    /// `Tensor` object can be looked up in `Op::compute`
    pub(crate) node_id: usize,
    /// Input arrays
    pub(crate) xs: Vec<OpInput<'v, T>>,
    /// Output arrays
    pub(crate) ys: ArrayVec<[Result<crate::ArrRepr<'v, T>, crate::op::ComputeException>; NUM_MAX_OUTPUT]>
}

const NUM_MAX_OUTPUT: usize = 16;

impl<'v, T: Float> OpComputeContext<'v, T> {
    /// Instantiates an `OpComputeContext` object.
    #[inline]
    pub(crate) fn new(node_id: usize, xs: Vec<OpInput<'v, T>>) -> Self {
        OpComputeContext {
            node_id,
            xs,
            ys: ArrayVec::<[_; NUM_MAX_OUTPUT]>::new(),
        }
    }

    /// Get a `i` th input array as a read-only view.
    ///
    /// Calling `input(i)` more than once causes panic.
    #[inline]
    pub fn input(&mut self, i: usize) -> NdArrayView<'v, T> {
        let x = match self.xs.get_mut(i) {
            Some(x) => x,
            None => panic!(
                "Bad op impl: input index out of range.",
            ),
        };
        match x {
            OpInput::RO(ref mut a) => match mem::replace(a, None) {
                Some(ret) => ret,
                None => panic!("Bad op impl: input({})/input_mut({}) cannot be called twice", i, i),
            },
            _ => {
                panic!("Bad op impl: cannot perform immutable borrowing for input({})", i);
            }
        }
    }

    /// Get a `i` th input array as a read-write view.
    ///
    /// Calling `input_mut(i)` more than once causes panic.
    #[inline]
    pub fn input_mut(&mut self, i: usize) -> NdArrayViewMut<'v, T> {
        let x = match self.xs.get_mut(i) {
            Some(x) => x,
            None => panic!(
                "Bad op impl: {}'s input doesn't exist.",
                i
            ),
        };
        match x {
            OpInput::RW(ref mut a) => match mem::replace(a, None) {
                Some(ret) => ret,
                None => panic!("Bad op impl: input({})/input_mut({}) cannot be called twice", i, i),
            },
            _ => {
                panic!("Bad op impl: cannot perform mutable borrowing for input({})", i);
            }
        }
    }

    /// Sets output arrays of an `Op`.
    ///
    /// Implementors of `Op::compute` must not forget to call this function, otherwise panic occurs.
    /// You can also use `push_output`.
    ///
    /// NOTE: the maximum number of output arrays are 16 for now.
    #[inline]
    pub fn set_output(
        &mut self,
        ys: ArrayVec<[Result<crate::ArrRepr<'v, T>, crate::op::ComputeException>; 16]>,
    ) {
        self.ys = ys;
    }

    /// Appends an nd-array to the back of the output list of the current op.
    ///
    /// Implementors of `Op::compute` must not forget to call this function, otherwise panic occurs.
    /// You can also use `set_output`.
    ///
    /// NOTE: the maximum number of output arrays are 16 for now.
    pub fn push_output(&mut self, y: Result<crate::ArrRepr<'v, T>, crate::op::ComputeException>) {
        match self.ys.try_push(y) {
            Ok(()) => {},
            _ => {
                panic!("Bad op impl: reached the maximum number of output arrays ({})", NUM_MAX_OUTPUT);
            }
        }
    }

    /// Returns a number of input arrays.
    #[inline]
    pub fn num_inputs(&self) -> usize {
        self.xs.len()
    }
}

#[derive(Debug)]
struct Node<'t, T: Float> {
    node: &'t Tensor<'t, T>,
    // the len matches the number of outputs of this node
    value_info_list: Vec<ValueInfo>,
    successors: Vec<&'t Tensor<'t, T>>,
    // initialized with the in-degree of base node.
    // when this is reduced to 0, `base` is ready to be evaluated.
    pending_count: Cell<usize>,
    scheduled_base: Cell<bool>,
}

use std::ops::Deref;

impl<'t, T: Float> Deref for Node<'t, T> {
    type Target = Tensor<'t, T>;
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        self.node
    }
}

impl<'t, T: Float> Node<'t, T> {
    #[inline]
    fn new(node: &'t Tensor<'t, T>, successor: Option<&'t Tensor<'t, T>>) -> Self {
        let mut successors = Vec::with_capacity(1);
        if let Some(suc) = successor {
            if !contains(successors.as_slice(), suc) {
                successors.push(suc);
            }
        }
        Node {
            node,
            successors,
            value_info_list: Vec::new(),
            pending_count: Cell::new(0),
            scheduled_base: Cell::new(false),
        }
    }

    #[inline]
    fn scheduled(&self) -> bool {
        self.scheduled_base.get()
    }

    #[inline]
    fn mark_scheduled(&self) {
        self.scheduled_base.set(true);
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
fn build_minimum_subgraph_from<'t, F>(targets: &[&'t Tensor<'t, F>]) -> Graph<'t, F>
where
    F: Float,
{
    let mut node_info = FxHashMap::<usize, Node<F>>::default();
    let mut sources = FxHashSet::default();
    let mut dfs_stack: Vec<&Tensor<_>> = Vec::with_capacity(128);

    // Initialize the graph and stack with `targets`
    for t in targets {
        let node = Node::new(t, None);
        if let Entry::Vacant(ent) = node_info.entry(t.id()) {
            let inserted = ent.insert(node);
            dfs_stack.push(inserted.node);
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
            match node_info.entry(child.id()) {
                Entry::Vacant(ent) => {
                    // initial visit
                    let inserted = ent.insert(Node::new(&child.val, Some(node)));
                    dfs_stack.push(inserted.node);
                }
                Entry::Occupied(mut ent) => {
                    let successors = &mut ent.get_mut().successors;
                    // ensuring no duplication in successors to handle the case like `y = add(x, x)`.
                    // linear search is enough here.
                    if !contains(successors.as_slice(), node) {
                        successors.push(node)
                    } else {
                        found_new_successor = false;
                    }
                }
            }
            if found_new_successor {
                node_info.get_mut(&node.id()).unwrap().increment_pending_count();
            }
        }
    }
    Graph { node_info, sources }
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

// type of ndarray
#[derive(Clone, Copy, PartialEq, Debug)]
enum ValueType {
    Owned,
    View,
    Empty,
}

#[derive(Clone, Debug)]
struct ValueInfo {
    ty: ValueType,
    // key to lookup output storage
    key: usize,
}

impl ValueInfo {
    #[inline]
    fn new(ty: ValueType, key: usize) -> Self {
        Self { ty, key }
    }
}

struct OpEvalResult<'view, T: Float> {
    node_id: usize,
    ys: crate::op::ComputeResults<'view, T>,
    called_compute: bool
}

struct OutputStorage<'view, F: Float> {
    inner: UnsafeCell<OutputStorageInner<'view, F>>,
}

struct OutputStorageInner<'view, F: Float> {
    owned: Vec<Option<NdArray<F>>>,
    borrowed: Vec<NdArrayView<'view, F>>,
}

impl<'tensor, 'view, 'lock, F: Float> OutputStorage<'view, F> {
    #[inline]
    fn new() -> Self {
        OutputStorage {
            inner: UnsafeCell::new(OutputStorageInner {
                owned: Vec::new(),
                borrowed: Vec::new(),
            }),
        }
    }

    #[inline]
    fn owned_mut(&self) -> &mut Vec<Option<NdArray<F>>> {
        unsafe { &mut (&mut *self.inner.get()).owned }
    }

    #[inline]
    fn owned(&self) -> &[Option<NdArray<F>>] {
        unsafe { &(&*self.inner.get()).owned }
    }

    #[inline]
    fn view_mut(&self) -> &mut Vec<NdArrayView<'view, F>> {
        unsafe { &mut (&mut *self.inner.get()).borrowed }
    }

    #[inline]
    fn view(&self) -> &[NdArrayView<'view, F>] {
        unsafe { &(&*self.inner.get()).borrowed }
    }
}

// map key is a tensor-id
struct LockGuardRegister<'lock, F: Float> {
    read_guards: UnsafeCell<FxHashMap<usize, Vec<Option<RwLockReadGuard<'lock, NdArray<F>>>>>>,
    write_guards: UnsafeCell<FxHashMap<usize, Vec<Option<RwLockWriteGuard<'lock, NdArray<F>>>>>>,
}

impl<'t, 'lock, F: Float> LockGuardRegister<'lock, F> {
    #[inline]
    fn init(&self, key: &'t Tensor<'t, F>) {
        unsafe {
            (&mut *self.write_guards.get()).insert(key.id(), crate::none_vec(key.inputs.len()));
            (&mut *self.read_guards.get()).insert(key.id(), crate::none_vec(key.inputs.len()));
        }
    }

    #[inline]
    fn new() -> Self {
        LockGuardRegister {
            read_guards: UnsafeCell::new(FxHashMap::default()),
            write_guards: UnsafeCell::new(FxHashMap::default()),
        }
    }

    #[inline]
    fn lock_rw(
        &self,
        node_id: usize,
        input_idx: usize,
        lock: &'lock Arc<RwLock<NdArray<F>>>,
    ) -> &mut RwLockWriteGuard<'lock, NdArray<F>> {
        unsafe {
            let got: &mut Vec<Option<_>> =
                (&mut *self.write_guards.get()).get_mut(&node_id).unwrap();
            got[input_idx] = Some(lock.write().unwrap());
            got[input_idx].as_mut().unwrap()
        }
    }

    #[inline]
    fn lock_write(
        &self,
        node_id: usize,
        input_idx: usize,
        lock: &'lock Arc<RwLock<NdArray<F>>>,
    ) -> &RwLockReadGuard<'lock, NdArray<F>> {
        unsafe {
            let got: &mut Vec<Option<_>> =
                (&mut *self.read_guards.get()).get_mut(&node_id).unwrap();
            got[input_idx] = Some(lock.read().unwrap());
            got[input_idx].as_ref().unwrap()
        }
    }

    #[inline]
    fn invalidate_input_guards_of(&self, node: &Tensor<'t, F>) {
        for (i, input) in node.inputs.iter().enumerate() {
            unsafe {
                if input.mut_usage {
                    mem::swap(
                        &mut (&mut *self.write_guards.get()).get_mut(&node.id()).unwrap()[i],
                        &mut None,
                    );
                } else {
                    mem::swap(
                        &mut (&mut *self.read_guards.get()).get_mut(&node.id()).unwrap()[i],
                        &mut None,
                    );
                }
            }
        }
    }
}

#[inline]
fn is_eval_target<'a, T>(node: &'a Tensor<T>, targets: &'a [&'a Tensor<'a, T>]) -> bool
where
    T: Float,
{
    for t in targets {
        // comparing node ids
        if node.id() == t.id() {
            return true;
        }
    }
    false
}

struct Graph<'t, F: Float> {
    // source nodes in this graph
    sources: FxHashSet<&'t Tensor<'t, F>>,
    // tensor -> Node(=its immediate successor nodes etc)
    node_info: FxHashMap<usize, Node<'t, F>>,
}

impl<'t, F: Float> Graph<'t, F> {
    #[inline]
    fn get(&self, key: &'t Tensor<'t, F>) -> &Node<'t, F> {
        self.node_info.get(&key.id()).unwrap()
    }
}

#[inline]
fn find_feed<'t, 'feeds, 'feed, F: Float>(
    feeds: &'feeds [Feed<'t, 'feed, F>],
    in_node: &'t Tensor<'t, F>,
) -> NdArrayView<'feeds, F> {
    // linear search is enough for feeds
    for feed in feeds {
        if feed.0 == in_node {
            let shape = feed.1.shape();
            in_node.validate_feed_shape(shape);
            return feed.1.view();
        }
    }
    panic!("Placeholder unfilled");
}

// Extract output arrays from `ys` and stores into `storage` (and `graph`).
fn extract_compute_results<'t, 'view, F: Float>(
    node: &'t Tensor<'t, F>,
    ys: crate::op::ComputeResults<'view, F>,
    storage: &OutputStorage<'view, F>,
    graph: &mut Graph<'t, F>,
) {
    let mut info_list = Vec::with_capacity(ys.len());
    for y in ys {
        let info = match y {
            Ok(crate::ArrRepr::Owned(val)) => {
                storage.owned_mut().push(Some(val));
                ValueInfo::new(ValueType::Owned, storage.owned().len() - 1)
            }
            Ok(crate::ArrRepr::View(val)) => {
                storage.view_mut().push(val);
                ValueInfo::new(ValueType::View, storage.view().len() - 1)
            }
            _ => ValueInfo::new(ValueType::Empty, /*dummy = */ 0),
        };
        info_list.push(info);
    }
    // info_list is stored in the `graph` object.
    graph.node_info.get_mut(&node.id()).unwrap().value_info_list = info_list;
}

use std::io::{self, Write};

/// Evaluates given symbolic tensors.
///
/// Each return value can be `None`;
/// for example, evaluation of `gradient_descent_ops::*`
/// would result in `None`.
///
/// NOTE: All the runtime errors are not reported by return values, but by "panic"
/// for convenience.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let ref a = ag::zeros(&[2]);
/// let ref b = ag::ones(&[2]);
///
/// // eval two tensors at once.
/// let evaluated = ag::eval(&[a, b], &[]);
/// assert_eq!(evaluated[0], Some(ndarray::arr1(&[0., 0.]).into_dyn()));
/// assert_eq!(evaluated[1], Some(ndarray::arr1(&[1., 1.]).into_dyn()));
/// ```
pub fn eval<'feed, 't, 'view, T, F>(
    tensors: &'t [&'t Tensor<'t, F>],
    feeds: &'feed [Feed<'t, 'view, F>],
) -> Vec<Option<NdArray<F>>>
where
    F: Float,
{
    // Storage in which compute results are stored.
    let storage = OutputStorage::new();
    // Storage for RAII guards of variable locks
    let lock_state = LockGuardRegister::new();
    let mut graph = build_minimum_subgraph_from(tensors);
    let (sender, receiver) = crossbeam_channel::unbounded();

    // Schedule source nodes.
    for &src in &graph.sources {
        let called_compute = src.requires_compute();
        let node_id = src.id();
        sender
            .send(OpEvalResult {
                called_compute,
                node_id,
                ys: if !src.requires_compute() {
                    ArrayVec::<[_; NUM_MAX_OUTPUT]>::new()
                } else {
                    let mut ctx = OpComputeContext::new(node_id, vec![]);
                    src.op.compute(&mut ctx);
                    if ctx.ys.is_empty() {
                        panic!("Bad op implementation: empty return value");
                    }
                    ctx.ys
                },
            })
            .unwrap();
    }

    let mut eval_remaining = tensors.len();

    // Main loop.
    loop {
        // Aggregate and register a compute result.
        let evaluated: &Node<F> = {
            let OpEvalResult { node_id, ys, called_compute } = receiver.recv().unwrap();
            let node = graph.node_info.get(&node_id).unwrap();
            lock_state.invalidate_input_guards_of(node);
            if called_compute {
                extract_compute_results(node, ys, &storage, &mut graph);
            }
            node
        };

        if is_eval_target(evaluated.node, tensors) {
            eval_remaining -= 1;
            if eval_remaining == 0 {
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
                for (i, (input, &in_idx)) in suc.inputs.iter().zip(&suc.input_indices).enumerate() {
                    let x = if input.is_placeholder {
                        OpInput::new(find_feed(feeds, &input.val))
                    } else if let Some(ref lock) = input.variable_array {
                        if input.mut_usage {
                            OpInput::new_mut(lock_state.lock_rw(suc.id(), i, lock).view_mut())
                        } else {
                            OpInput::new(lock_state.lock_write(suc.id(), i, lock).view())
                        }
                    } else if let Some(ref arr) = input.get_constant_array() {
                        OpInput::new(arr.view())
                    } else {
                        // Search the output of other nodes
                        let vi = &graph.node_info.get(&input.id()).unwrap().value_info_list[in_idx];
                        match vi.ty {
                            ValueType::Owned => {
                                OpInput::new(storage.owned()[vi.key].as_ref().unwrap().view())
                            }
                            ValueType::View => OpInput::new(storage.view()[vi.key].clone()),
                            ValueType::Empty => {
                                panic!(
                                    "Attempting to use {}'s output which is empty.",
                                    input.op.name()
                                );
                            }
                        }
                    };
                    xs.push(x);
                }

                let op = &suc.op;
                let node_id = suc.id();
                crate::rayon::scope(|s| {
                    s.spawn(|_| {
                        // run compute
                        let mut ctx = OpComputeContext::new(node_id, xs);
                        op.compute(&mut ctx);
                        if ctx.ys.is_empty() {
                            panic!("Bad op implementation: empty return value");
                        }
                        sender
                            .send(OpEvalResult {
                                called_compute: true,
                                node_id,
                                ys: ctx.ys
                            })
                            .unwrap();
                    })
                });
            }
        }
    } // loop end

    // Aggregate return values
    let mut ret = Vec::with_capacity(tensors.len());
    for t in tensors {
        let arr = if let Some(per) = t.clone_persistent_array() {
            Some(per)
        } else if t.is_placeholder {
            Some(find_feed(feeds, t).to_owned())
        } else {
            let info = &graph.node_info.get(&t.id()).unwrap().value_info_list[0];
            if ValueType::Owned == info.ty {
                mem::replace(&mut storage.owned_mut()[info.key], None)
            } else if ValueType::View == info.ty {
                Some(storage.view()[info.key].to_owned())
            } else {
                None
            }
        };
        ret.push(arr);
    }
    ret
}

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
