use crate::ndarray_ext::{NdArray, NdArrayView, NdArrayViewMut};
use crate::op;
use crate::tensor::Tensor;
use crate::Float;
use crate::FxHashMap;
use crate::hashbrown::hash_map::Entry;
use ndarray;
use std::mem;
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::cell::Cell;
use crossbeam::crossbeam_channel;
use std::cell::UnsafeCell;

/// Helper structure for batched evaluation.
///
/// Use this in case `ag::eval` doesn't help.
///
/// ```
/// extern crate autograd as ag;
/// extern crate ndarray as nd;
///
/// let ref a = ag::placeholder(&[]);
/// let ref x = a + a;
/// let ref y = a * a;
/// let ref z = a / a;
///
/// ag::Eval::new()
///     .push(&y)
///     .extend(&[y, z])
///     .run(&[ag::Feed(a, nd::arr0(2.).into_dyn().view())]);  // Do eval
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
    RO(Option<NdArrayView<'v, T>>),  // Read-only view
    RW(Option<NdArrayViewMut<'v, T>>),  // Read-write view
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

pub struct OpComputeContext<'v, T: Float> {
    pub(crate) node_id: usize,
    pub(crate) xs: Vec<OpInput<'v, T>>,
    pub(crate) ys: Option<Vec<Result<crate::ArrRepr<'v, T>, crate::op::ComputeException>>>,
}

impl<'v, T: Float> OpComputeContext<'v, T> {
    #[inline]
    pub fn input(&mut self, i: usize) -> NdArrayView<'v, T> {
        match &mut self.xs[i] {
            OpInput::RO(ref mut a) => {
                match mem::replace(a, None) {
                    Some(ret) => ret,
                    None => panic!("Bad op impl: OpComputeContext::input({}) cannot be called twice", i)
                }
            }
            _ => {
                panic!("Bad op impl: cannot perform immutable borrowing for input({})", i);
            }
        }
    }

    #[inline]
    pub fn input_mut(&mut self, i: usize) -> NdArrayViewMut<'v, T> {
        match self.xs[i] {
            OpInput::RW(ref mut a) => {
                match mem::replace(a, None) {
                    Some(ret) => ret,
                    None => panic!("Bad op impl: OpComputeContext::input_mut({}) cannot be called twice", i)
                }
            },
            _ => {
                panic!("Bad op impl: cannot perform mutable borrowing for input({})", i);
            }
        }
    }

    #[inline]
    pub(crate) fn new(node_id: usize, xs: Vec<OpInput<'v, T>>) -> Self {
        OpComputeContext {
            node_id,
            xs,
            ys: None
        }
    }

    #[inline]
    pub fn set_output(&mut self,
                      ys: Vec<Result<crate::ArrRepr<'v, T>, crate::op::ComputeException>>) {
        self.ys = Some(ys)
    }

    #[inline]
    pub fn num_inputs(&self) -> usize {
        self.xs.len()
    }
}

#[derive(Debug)]
struct EdgeInfo<'node, T: Float> {
    node: NodeInfo<'node, T>,
    successors: Vec<&'node Tensor<T>>,
    // initialized with the number of the immediate predecessors.
    // When this is reduced to 0, `node` is ready to be evaluated.
    pending_count: Cell<usize>,
    scheduled: Cell<bool>
}

//impl<'node, T: Float> EdgeInfo<'node, T> {
//    #[inline]
//    fn new(node: &'node Tensor<T>, successors: Vec<&'node Tensor<T>>) -> Self {
//        EdgeInfo {
//            node,
//            successors,
//            pending_count: Cell::new(node.inputs.len()),
//            scheduled: Cell::new(false)
//        }
//    }
//
//    #[inline]
//    fn scheduled(&self) -> bool {
//        self.scheduled.get()
//    }
//
//    #[inline]
//    fn mark_scheduled(&self) {
//        self.scheduled.set(true);
//    }
//
//    #[inline]
//    fn decrement_pending_count(&self) {
//        self.pending_count.set(self.pending_count.get().saturating_sub(1));
//    }
//
//    #[inline]
//    fn ready(&self) -> bool {
//        self.pending_count.get() == 0
//    }
//}

#[derive(Debug)]
struct NodeInfo<'a, T: Float> {
    inner: &'a Tensor<T>,
    successors: Vec<&'a Tensor<T>>,
    // initialized with the number of the immediate predecessors.
    // When this is reduced to 0, `node` is ready to be evaluated.
    pending_count: Cell<usize>,
    scheduled: Cell<bool>
}

use std::ops::{Deref, DerefMut};

impl<'a, T: Float> Deref for NodeInfo<'a, T> {
    type Target = Tensor<T>;
    fn deref(&self) -> &Self::Target {
        self.inner
    }
}

impl<'a, T: Float> NodeInfo<'a, T> {
    #[inline]
    fn new(inner: &'a Tensor<T>, successors: Vec<&'a Tensor<T>>) -> Self {
        NodeInfo {
            inner,
            pending_count: Cell::new(0),
            successors,
            scheduled: Cell::new(false)
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
        self.pending_count.set(self.pending_count.get().saturating_sub(1));
    }

    #[inline]
    fn ready(&self) -> bool {
        self.pending_count.get() == 0
    }
}

// Builds a subgraph consisting of nodes that are reachable from `tensors`.
// Returns: edge_map, source_nodes
fn build_minimum_subgraph_from<K, T>(
    tensors: &[K],
) -> (FxHashMap<usize, NodeInfo<T>>, Vec<&Tensor<T>>)
    where
        K: AsRef<Tensor<T>>,
        T: Float
{
    // mapping of node_id -> EdgeInfo
    let mut edge_map = FxHashMap::<usize, NodeInfo<T>>::default();
    // set of source nodes
    let mut source_nodes = Vec::with_capacity(16);

    // pointers are dropped in this function.
    let mut dfs_stack: Vec<*mut NodeInfo<_>> = Vec::with_capacity(128);

    for t in tensors {
        let t = NodeInfo::new(t.as_ref(), vec![]);
        if let Entry::Vacant(ent) = edge_map.entry(t.tensor_id()) {
            let inserted = ent.insert(t);
            dfs_stack.push(inserted as *mut _);
        } else {
            panic!("Duplication in the given evaluation target was detected.");
        }
    }

    unsafe {
        while let Some(node) = dfs_stack.pop() {
            let node = &mut *node;
            if node.is_source() && !contains(&source_nodes, node.inner){
                source_nodes.push(node);
            }
            for child in &node.inputs {
                match edge_map.entry(child.tensor_id()) {
                    Entry::Vacant(ent) => {  // initial visit
                        let inserted = ent.insert(NodeInfo::new(&child.val, vec![node]));
                        dfs_stack.push(inserted as *mut _);
                        node.increment_pending_count()
                    }
                    Entry::Occupied(mut ent) => {
                        let successors = &mut ent.get_mut().successors;
                        // ensuring no duplication in successors to handle the case like `y = add(x, x)`.
                        // linear search is enough here.
                        if !contains(successors.as_slice(), node.inner) { // Make a edge
                            successors.push(node.inner);
                            node.increment_pending_count();
                        }
                    }
                }
            }
        }
        (edge_map, source_nodes)
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

struct OpEvalResult<'v, T: Float> {
    node_id: usize,
    compute_was_called: bool,
    ys: crate::op::ComputeResults<'v, T>,
}

#[inline]
fn is_eval_target<A, T>(
    node: &Tensor<T>,
    targets: &[A],
) -> bool
    where
        A: AsRef<Tensor<T>>,
        T: Float,
{
    for t in targets {
        if node == t.as_ref() {  // comparing node ids
            return true;
        }
    }
    false
}

pub fn eval<'slice, 'node, 'feed, K, T>(
    tensors: &'node [K],
    feeds: &'slice [Feed<'node, 'feed, T>],
) -> Vec<Option<NdArray<T>>>
    where
        K: AsRef<Tensor<T>>,
        T: Float,
{
    // edge_map: node-id -> EdgeInfo
    let (edge_map, sources) = build_minimum_subgraph_from(tensors);

    for (k, v) in edge_map.iter() {
        println!("{:?}", v);
    }

    let mut output_info_storage = FxHashMap::<usize, NodeMetadata<T>>::default();

    // Storage in which compute results are stored.
    let mut owned_storage = UnsafeCell::new(Vec::<NdArray<T>>::with_capacity(100));
    let mut read_guards_map = UnsafeCell::new(FxHashMap::<usize, Vec<Option<RwLockReadGuard<_>>>>::default());
    let mut write_guards_map = UnsafeCell::new(FxHashMap::<usize, Vec<Option<RwLockWriteGuard<_>>>>::default());

    {
        // Views of owned arrays whose lifetime is shorter than owned ones
        let mut view_storage = Vec::<NdArrayView<T>>::new();

        let (sender, receiver) = crossbeam_channel::bounded(16);

        for src in sources {  // source: placeholder, constant, variable, generator.
            let node_id = src.tensor_id();
            let is_basic = src.is_placeholder || src.has_persistent_array;
            sender.send(OpEvalResult {
                node_id,
                compute_was_called: !is_basic,
                ys: if is_basic { vec![] } else {
                    let mut ctx = OpComputeContext::new(node_id, vec![]);
                    src.op.compute(&mut ctx);
                    ctx.ys.unwrap()
                }
            });
        }

        let mut targets_remaining = tensors.len();

        loop {
            // Aggregate and register a compute result.
            let evaluated: &NodeInfo<T> = unsafe {
                let OpEvalResult { node_id, compute_was_called, ys } = receiver.recv().unwrap();
                for (i, input) in edge_map.get(&node_id).unwrap().inner.inputs.iter().enumerate() {
                    if input.mut_usage {
                        mem::swap(&mut (&mut *write_guards_map.get()).get_mut(&node_id).unwrap()[i], &mut None);
                    } else {
                        mem::swap(&mut (&mut *read_guards_map.get()).get_mut(&node_id).unwrap()[i], &mut None);
                    }
                }
                if compute_was_called {
                    let mut info_list = Vec::with_capacity(ys.len());
                    let mut contains_no_output = false;
                    for y in ys {
                        match y {
                            Ok(crate::ArrRepr::Owned(val)) => {
                                info_list.push(ValueInfo::new(
                                    ValueType::Owned,
                                    (&*owned_storage.get()).len(),
                                ));
                                (&mut *owned_storage.get()).push(val);
                            }
                            Ok(crate::ArrRepr::View(val)) => {
                                info_list.push(ValueInfo::new(ValueType::View, view_storage.len()));
                                view_storage.push(val);
                            }
                            _ => {
                                info_list.push(ValueInfo::new(ValueType::Empty, 0));
                                contains_no_output = true;
                            }
                        }
                    }
                    output_info_storage.insert(
                        node_id,
                        NodeMetadata { info_list, contains_no_output }
                    );
                }
                edge_map.get(&node_id).unwrap()
            };

            if is_eval_target(evaluated.inner, tensors) {
                targets_remaining -= 1;
                if targets_remaining == 0 {
                    break;  // exit the main loop as all the target tensors were evaluated.
                }
            }

            // Try to schedule the immediate successors of the evaluated node.
            println!("\nevaluated {:?}", &evaluated);
            for &suc in &evaluated.successors {
                println!("suc {:?}", &suc);
                let suc_id = suc.tensor_id();
                let suc_info = edge_map.get(&suc_id).unwrap();
                suc_info.decrement_pending_count();

                if !suc_info.scheduled() && suc_info.ready() {
                    suc_info.mark_scheduled();
                    println!("marked: {:?}", suc);
                    // Prepare and send inputs to executor.
                    unsafe {
                        (&mut *write_guards_map.get()).insert(suc_id, crate::none_vec(suc.inputs.len()));
                        (&mut *read_guards_map.get()).insert(suc_id, crate::none_vec(suc.inputs.len()));
                    }

                    println!("initialized read_guards_map");

                    let mut xs = Vec::with_capacity(suc.inputs.len());

                    // Aggregate inputs for `in_node`
                    for (i, (in_node, &in_idx)) in suc.inputs.iter().zip(&suc.input_indices).enumerate() {
                        println!("aggregating: {:?}", in_node.val);
                        if in_node.is_placeholder {
                            for feed in feeds {  // linear search is enough for feeds
                                if Arc::ptr_eq(feed.0, &in_node.val) {
                                    let clone = feed.1.view();
                                    if !in_node.known_shape.as_ref().unwrap().validate(clone.shape()) {
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
                                    let mut got_mut: &mut Vec<_> = (&mut *write_guards_map.get()).get_mut(&suc_id).unwrap();
                                    got_mut[i] = Some(lock.write().unwrap());
                                    xs.push(OpInput::new_mut(got_mut[i].as_mut().unwrap().view_mut()));
                                } else {
                                    let mut got_mut: &mut Vec<_> = (&mut *read_guards_map.get()).get_mut(&suc_id).unwrap();
                                    got_mut[i] = Some(lock.read().unwrap());
                                    xs.push(OpInput::new(got_mut[i].as_ref().unwrap().view()));
                                }
                            }
                        } else if let Some(ref arr) = in_node.get_constant_array() {
                            xs.push(OpInput::new(arr.view()));
                        } else {
                            // Search the output of other nodes
                            let info: &ValueInfo<T> =
                                &output_info_storage.get(&in_node.tensor_id()).unwrap().info_list[in_idx];
                            match info.ty {
                                ValueType::Owned => unsafe {
                                    xs.push(OpInput::new((&*owned_storage.get())[info.key].view()));
                                }
                                ValueType::View => { // Clone the view
                                    xs.push(OpInput::new(view_storage[info.key].clone()));
                                }
                                ValueType::Empty => {
                                    panic!("{}'s output, which is empty, was fed to {}", in_node.op.name(), suc.op.name());
                                }
                            }
                        }
                    }

                    crate::rayon::scope(|s| { s.spawn(|_| {
                        // run compute
                        let mut ctx = OpComputeContext::new(suc_id, xs);
                        suc.op.compute(&mut ctx);
                        sender.send(OpEvalResult {
                            node_id: suc_id,
                            compute_was_called: true,
                            ys: ctx.ys.expect("Bad op implementation: empty return value")
                        });
                    })});
                }
            }
        }

        // process array views
        for t in tensors {
            let t = t.as_ref();
            if !t.is_placeholder && !t.has_persistent_array {
                let info: &mut ValueInfo<T> =
                    &mut output_info_storage.get_mut(&t.tensor_id()).unwrap().info_list[0];
                if let ValueType::View = info.ty {
                    info.value = Some(view_storage[info.key].to_owned());
                }
            }
        }

    }

    let owned_storage = unsafe { &mut *owned_storage.get() };
    for t in tensors {
        let t = t.as_ref();
        if !t.is_placeholder && !t.has_persistent_array {
            let info: &mut ValueInfo<T> = &mut output_info_storage.get_mut(&t.tensor_id()).unwrap().info_list[0];
            if let ValueType::Owned = info.ty {
                info.value = Some(owned_storage[info.key].to_owned());
            }
        }
    }

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
            mem::replace(
                &mut output_info_storage.get_mut(&t.tensor_id()).unwrap().info_list[0].value,
                None,
            )
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
struct ValueInfo<T: Float> {
    ty: ValueType,
    // key to lookup the value
    key: usize,
    // Owned array
    value: Option<NdArray<T>>,
}

impl<T: Float> ValueInfo<T> {
    #[inline]
    fn new(ty: ValueType, key: usize) -> ValueInfo<T> {
        ValueInfo {
            ty,
            key,
            value: None,
        }
    }
}

struct NodeMetadata<T: Float> {
    info_list: Vec<ValueInfo<T>>, // ys
    contains_no_output: bool,
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
