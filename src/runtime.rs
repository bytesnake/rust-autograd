//! Defining things related to evaluation of `ag::Tensor`.
use crate::arrayvec::ArrayVec;
use crate::ndarray_ext::{NdArray, NdArrayView};
use crate::op::{OpComputeContext, OpInput, NUM_MAX_OUTPUT};
use crate::tensor::{Tensor, TensorInternal, PersistentArray};
use crate::{hashbrown::hash_map::Entry, FxHashMap, FxHashSet};
use crate::{Float, Graph};
use crossbeam::crossbeam_channel;
use std::cell::Cell;
use std::cell::UnsafeCell;
use std::mem;
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

/// Helper structure for batched evaluation.
///
/// Use this in case `ag::eval` doesn't help.
///
/// ```
/// use autograd as ag;
/// use ndarray;
///
/// ag::with(|g| {
///    let a = g.placeholder(&[]);
///    let x = a + a;
///    let y = a * a;
///    let z = a / a;
///
///    ag::Eval::new(g)
///        .push(&x)
///        .extend(&[y, z])
///        .feed(&[a.given(ndarray::arr0(2.).view())])
///        .run();  // Do eval
///
///    });
/// ```
pub struct Eval<'v, 'f, 't, 's: 't, F: Float> {
    scope: &'s Graph<F>,
    buf: Vec<Tensor<'t, 's, F>>,
    feeds: Option<&'f [crate::runtime::Feed<'v, F>]>,
}

impl<'f, 't, 'v, 's: 't, F: Float> Eval<'v, 'f, 't, 's, F> {
    #[inline]
    /// Instantiates a new evaluation session.
    pub fn new(scope: &'s Graph<F>) -> Self {
        Eval {
            feeds: None,
            scope,
            buf: Vec::new(),
        }
    }

    #[inline]
    /// Appends a tensor to the back of the evaluation targets.
    pub fn push<A>(&mut self, x: A) -> &mut Self
        where
            A: AsRef<Tensor<'t, 's, F>>,
    {
        self.buf.push(*x.as_ref());
        self
    }

    /// `feeds` is a sequence of `(placeholder-tensor, its value)`
    pub fn feed(&mut self, feeds: &'f [crate::Feed<'v, F>]) -> &mut Self {
        self.feeds = Some(feeds);
        self
    }

    #[inline]
    /// Extends the evaluation targets with `xs`.
    pub fn extend<A>(&mut self, xs: &'t [A]) -> &mut Self
        where
            A: AsRef<Tensor<'t, 's, F>>,
    {
        self.buf.extend(xs.iter().map(|x| *x.as_ref()));
        self
    }

    #[inline]
    /// Evaluates the buffered tensors.
    pub fn run(&'t self) -> Vec<Option<NdArray<F>>> {
        self.scope
            .eval(self.buf.as_slice(), self.feeds.unwrap_or(&[]))
    }
}

/// Links a placeholder tensor and its value at run-time.
///
/// Use `Tensor::given` or `Feed::new` to instanciate.
/// Ensure that this is passed to `ag::Eval`, `ag::eval` or `Tensor::eval`.
///
/// ```
/// use ndarray::array;
/// use autograd as ag;
///
/// ag::with(|g| {
///     let x = g.placeholder(&[2]);
///
///     // Fills the placeholder with an ArrayView, then eval.
///     let value = array![1., 1.];
///     x.eval(&[x.given(value.view())]);
/// });
/// ```
pub struct Feed<'feed, T: Float> {
    /// The id of the placeholder tensor
    placeholder_id: usize,
    /// A run-time value of the placeholder
    value: NdArrayView<'feed, T>,
}

impl<'feed, F: Float> Feed<'feed, F> {
    /// Substantially same as `Tensor::given(value)`.
    #[inline]
    pub fn new(placeholder_id: usize, value: NdArrayView<'feed, F>) -> Self {
        Feed {
            placeholder_id,
            value,
        }
    }
}

#[derive(Debug)]
struct NodeInfo<'t, T: Float> {
    node: &'t TensorInternal<T>,
    // the len matches the number of outputs of this node
    value_info_list: Vec<ValueInfo>,
}

struct NodeInfoAsync<'t, F: Float> {
    node: &'t TensorInternal<F>,
    // the len matches the number of outputs of this node.
    value_info_list: Vec<ValueInfo>,
    successors: Vec<&'t TensorInternal<F>>,
    in_persistent_arrays: Vec<PersistentArray<'t, F>>,
    // idx to lookup evaluation stats.
    target_idx: Option<usize>,
    // initialized with the in-degree of `node`;
    // when this is reduced to 0, `node` is ready to be evaluated.
    pending_count: Cell<usize>,
    scheduled: Cell<bool>,
}

use std::ops::Deref;

impl<'t, T: Float> Deref for NodeInfoAsync<'t, T> {
    type Target = TensorInternal<T>;
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        self.node
    }
}

impl<'t, 's: 't, F: Float> NodeInfoAsync<'t, F> {
    fn new(
        node: &'t TensorInternal<F>,
        successor: Option<&'t TensorInternal<F>>,
        g: &'s Graph<F>,
        target_idx: Option<usize>
    ) -> Self
    {
        let mut successors = Vec::new();
        if let Some(suc) = successor {
            if !contains(successors.as_slice(), suc) {
                successors.push(suc);
            }
        }

        // Collect input arrays using graph beforehand since `Graph` can't shared between threads.
        let mut persistent_input_arrays = Vec::with_capacity(node.in_edges.len());
        for x in &node.in_edges {
            persistent_input_arrays.push(x.get_inner(g).get_persistent_array());
        }
        NodeInfoAsync {
            node,
            successors,
            in_persistent_arrays: persistent_input_arrays,
            target_idx,
            value_info_list: Vec::new(),
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
fn build_stateful_subgraph_from<'t, 's: 't, F, A>(
    targets: &'t [A],
    graph: &'s Graph<F>,
) -> SubGraph<'t, F>
    where
        F: Float,
        A: AsRef<Tensor<'t, 's, F>> + Copy,
{
    let mut node_info = FxHashMap::<usize, NodeInfoAsync<F>>::default();
    let mut sources = FxHashSet::default();
    let mut dfs_stack: Vec<&TensorInternal<_>> = Vec::with_capacity(128);

    // Initialize the graph and stack with `targets`
    for (i, t) in targets.iter().enumerate() {
        let t = t.as_ref();
        let node = NodeInfoAsync::new(t.tensor, None, graph, Some(i));
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
        for child in &node.in_edges {
            let mut found_new_successor = true;
            match node_info.entry(child.get(graph).id()) {
                Entry::Vacant(ent) => {
                    // initial visit
                    let inserted = ent.insert(NodeInfoAsync::new(child.get_inner(graph), Some(node), graph, None));
                    dfs_stack.push(inserted.node);
                }
                Entry::Occupied(mut ent) => {
                    let successors = &mut ent.get_mut().successors;
                    // ensuring no duplication in successors to handle the case like `y = add(x, x)`.
                    if !contains(successors.as_slice(), node) {
                        successors.push(node)
                    } else {
                        found_new_successor = false;
                    }
                }
            }
            if found_new_successor {
                node_info
                    .get_mut(&node.id())
                    .unwrap()
                    .increment_pending_count();
            }
        }
    }
    SubGraph { map: node_info, sources, status: CompletionStatus::new(targets) }
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
    // key to lookup output
    key: usize,
}

impl ValueInfo {
    #[inline]
    fn new(ty: ValueType, key: usize) -> Self {
        ValueInfo { ty, key }
    }
}

struct OpEvalResult<'tensor, 'view, T: Float> {
    tensor: &'tensor TensorInternal<T>,
    ys: crate::op::ComputeResults<'view, T>,
}

struct ViewStorage<'view, F: Float> {
    inner: Vec<NdArrayView<'view, F>>,
}

struct ValueStorage<F: Float> {
    inner: Vec<Option<NdArray<F>>>,
}

impl<'view, F: Float> ViewStorage<'view, F> {
    #[inline]
    unsafe fn get_untracked_view(&self, key: usize) -> NdArrayView<'view, F> {
        let ptr: *const NdArrayView<'view, F> = &*&self.inner[key];
        (*ptr).clone()
    }

    #[inline]
    // Returns the inserted position.
    fn push(&mut self, view: NdArrayView<'view, F>) -> usize {
        self.inner.push(view);
        self.inner.len() - 1
    }
}

impl<'view, F: Float> ValueStorage<F> {
    #[inline]
    unsafe fn get_untracked_view(&self, key: usize) -> NdArrayView<'view, F> {
        let ptr: *const NdArray<F> = &*self.inner[key].as_ref().unwrap();
        (*ptr).view()
    }

    #[inline]
    // Returns the inserted position.
    fn push(&mut self, value: NdArray<F>) -> usize {
        self.inner.push(Some(value));
        self.inner.len() - 1
    }
}

impl<'view, F: Float> ViewStorage<'view, F> {
    #[inline]
    fn get_mut(&mut self) -> &mut Vec<NdArrayView<'view, F>> {
//        unsafe { &mut (&mut *self.inner.get()).borrowed }
        &mut self.inner
    }

    #[inline]
    fn view(&self) -> &[NdArrayView<'view, F>] {
//        unsafe { &(&*self.inner.get()).borrowed }
        self.inner.as_slice()
    }
}

impl<F: Float> ValueStorage<F> {

    #[inline]
    fn get_mut(&mut self) -> &mut Vec<Option<NdArray<F>>> {
//        unsafe { &mut (&mut *self.inner.get()).owned }
        &mut self.inner
    }

    #[inline]
    fn get(&self) -> &[Option<NdArray<F>>] {
//        unsafe { &(&*self.inner.get()).owned }
        &self.inner
    }
}


struct OutputStorage<'view, F: Float> {
    inner: UnsafeCell<OutputStorageInner<'view, F>>,
//    inner: OutputStorageInner<'view, F>,
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
            })
//            inner: OutputStorageInner {
//                owned: Vec::new(),
//                borrowed: Vec::new(),
//            }
        }
    }

    #[inline]
    fn owned_mut(&self) -> &mut Vec<Option<NdArray<F>>> {
        unsafe { &mut (&mut *self.inner.get()).owned }
//        &mut self.inner.owned
    }

    #[inline]
    fn owned(&self) -> &[Option<NdArray<F>>] {
        unsafe { &(&*self.inner.get()).owned }
//        &self.inner.owned
    }

    #[inline]
    fn view_mut(&self) -> &mut Vec<NdArrayView<'view, F>> {
        unsafe { &mut (&mut *self.inner.get()).borrowed }
//        &mut self.inner.borrowed
    }

    #[inline]
    fn view(&self) -> &[NdArrayView<'view, F>] {
        unsafe { &(&*self.inner.get()).borrowed }
//        &self.inner.borrowed
    }
}

// map key is a tensor-id
struct LockGuardRegistry<'lock, F: Float> {
    read_guards: UnsafeCell<FxHashMap<usize, Vec<Option<RwLockReadGuard<'lock, NdArray<F>>>>>>,
    write_guards: UnsafeCell<FxHashMap<usize, Vec<Option<RwLockWriteGuard<'lock, NdArray<F>>>>>>,
//    read_guards: FxHashMap<usize, Vec<Option<RwLockReadGuard<'lock, NdArray<F>>>>>,
//    write_guards: FxHashMap<usize, Vec<Option<RwLockWriteGuard<'lock, NdArray<F>>>>>,
}

impl<'t, 'lock, F: Float> LockGuardRegistry<'lock, F> {
    #[inline]
    fn init(&self, key: &'t TensorInternal<F>) {
        unsafe {
            (&mut *self.write_guards.get()).insert(key.id(), crate::none_vec(key.in_edges.len()));
            (&mut *self.read_guards.get()).insert(key.id(), crate::none_vec(key.in_edges.len()));
        }
//        self.write_guards.insert(key.id(), crate::none_vec(key.in_edges.len()));
//        self.read_guards.insert(key.id(), crate::none_vec(key.in_edges.len()));
    }

    #[inline]
    fn new() -> Self {
        LockGuardRegistry {
            read_guards: UnsafeCell::new(FxHashMap::default()),
            write_guards: UnsafeCell::new(FxHashMap::default()),
//            read_guards: FxHashMap::default(),
//            write_guards: FxHashMap::default(),
        }
    }

    #[inline]
    fn lock_rw(
        &self,
        node_id: usize,
        input_idx: usize,
        lock: &'lock RwLock<NdArray<F>>,
    ) -> &mut RwLockWriteGuard<'lock, NdArray<F>> {
        unsafe {
            let got: &mut Vec<Option<_>> = (&mut *self.write_guards.get()).get_mut(&node_id).unwrap();
//            let got: &mut Vec<Option<_>> = self.write_guards.get_mut(&node_id).unwrap();
            got[input_idx] = Some(lock.write().unwrap());
            got[input_idx].as_mut().unwrap()
        }
    }

    #[inline]
    fn lock_write(
        &self,
        node_id: usize,
        input_idx: usize,
        lock: &'lock RwLock<NdArray<F>>,
    ) -> &RwLockReadGuard<'lock, NdArray<F>> {
        unsafe {
            let got: &mut Vec<Option<_>> = (&mut *self.read_guards.get()).get_mut(&node_id).unwrap();
//            let got: &mut Vec<Option<_>> = self.read_guards.get_mut(&node_id).unwrap();
            got[input_idx] = Some(lock.read().unwrap());
            got[input_idx].as_ref().unwrap()
        }
    }

    #[inline]
    fn deregister_input_guards_of(&self, ten: &TensorInternal<F>) {
        for (i, input) in ten.in_edges.iter().enumerate() {
            unsafe {
                if input.mut_usage {
                    mem::swap(
                        &mut (&mut *self.write_guards.get()).get_mut(&ten.id()).unwrap()[i],
//                        &mut self.write_guards.get_mut(&ten.id()).unwrap()[i],
                        &mut None,
                    );
                } else {
                    mem::swap(
                        &mut (&mut *self.read_guards.get()).get_mut(&ten.id()).unwrap()[i],
//                        &mut self.read_guards.get_mut(&ten.id()).unwrap()[i],
                        &mut None,
                    );
                }
            }
        }
    }
}

struct SubGraph<'t, F: Float> {
    // collection of source nodes in this graph
    sources: FxHashSet<&'t TensorInternal<F>>,
    // tensor_id -> NodeInfo
    map: FxHashMap<usize, NodeInfoAsync<'t, F>>,
    status: CompletionStatus
}

impl<'t, F: Float> SubGraph<'t, F> {
    #[inline]
    fn get(&self, key: &'t TensorInternal<F>) -> &NodeInfoAsync<'t, F> {
        self.map.get(&key.id()).unwrap()
    }
}

fn validate_feed_shapes<F: Float>(feeds: &[Feed<F>], g: &Graph<F>) {
    for feed in feeds {
        let shape = feed.value.shape();
        g.access_node(feed.placeholder_id)
            .validate_feed_shape(shape);
    }
}

#[inline]
fn retrieve_feed<'t, 'feeds, 'feed, F: Float>(
    feeds: &'feeds [Feed<'feed, F>],
    in_node_id: usize,
) -> NdArrayView<'feeds, F> {
    // linear search is tolerable for feeds in most cases.
    for feed in feeds {
        if feed.placeholder_id == in_node_id {
            return feed.value.view();
        }
    }
    panic!("Placeholder unfilled");
}

// Extract output arrays from `ys` and stores into `storage` (and `node`).
fn extract_compute_results<'t, 'view, F: Float>(
    results: crate::op::ComputeResults<'view, F>,
    storage: &OutputStorage<'view, F>,
    node: &'t TensorInternal<F>,
) -> NodeInfo<'t, F> {
    let mut value_info_list = Vec::with_capacity(results.len());
    for y in results {
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
        value_info_list.push(info);
    }
    NodeInfo {
        node,
        value_info_list,
    }
}

// Extract output arrays from `ys` and stores into `storage` (and `graph`).
fn extract_compute_results_async<'t, 'view, F: Float>(
    ys: crate::op::ComputeResults<'view, F>,
    value_storage: &mut ValueStorage<F>,
    view_storage: &mut ViewStorage<'view, F>,
//    view_storage_ref: &mut ViewStorageRef<'view, F>,
    node: &mut NodeInfoAsync<'t, F>,
) {
    let mut info_list = Vec::with_capacity(ys.len());
    for y in ys {
        let info = match y {
            Ok(crate::ArrRepr::Owned(val)) => {
                let key = value_storage.push(val);
                ValueInfo::new(ValueType::Owned, key)
            }
            Ok(crate::ArrRepr::View(val)) => {
                let key = view_storage.push(val);
                ValueInfo::new(ValueType::View, key)
            }
            _ => ValueInfo::new(ValueType::Empty, /*dummy=*/ usize::default()),
        };
        info_list.push(info);
    }
    // value_info_list is stored in the Node object.
    node.value_info_list = info_list;
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum NodeStatus {
    Completed,
    NotYet
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum GraphStatus {
    Completed,
    NotYet
}

struct CompletionStatus {
    // id -> status
    target_statuses: Vec<(usize, NodeStatus)>,
    whole_status: GraphStatus,
    targets_remaining: usize,
}

impl CompletionStatus {

    #[inline]
    fn new<'tensor, 'scope, A, F>(targets: &'tensor [A]) -> Self
        where
            F: Float,
            A: AsRef<Tensor<'tensor, 'scope, F>> + Copy,
    {
        let num_targets = targets.len();
        let mut target_statuses = Vec::with_capacity(num_targets);
        for t in targets {
            target_statuses.push((t.as_ref().id(), NodeStatus::NotYet))
        }
        Self {
            targets_remaining: num_targets,
            whole_status: GraphStatus::NotYet,
            target_statuses,
        }
    }

    // updates targets_remaining if necessary and returns the status
    #[inline]
    fn maybe_update_with<F: Float>(&mut self, evaluated: &NodeInfoAsync<F>) -> GraphStatus {
        if let Some(idx) = evaluated.target_idx {  // if `evaluated` is the evaluation target..
            let mut slot = self.target_statuses[idx];
            if slot.1 == NodeStatus::NotYet {
                slot.1 = NodeStatus::Completed;
                // saturated subtraction is not need here.
                self.targets_remaining -= 1;
                if self.targets_remaining == 0 {
                    self.whole_status = GraphStatus::Completed;
                }
            }
        }
        self.whole_status
    }
}

struct ArrayViewProps {

}

impl<F: Float> Graph<F> {
    #[allow(dead_code)]
    // FIXME: too slow
    pub fn eval<'feed, 'tensor, 'scope: 'tensor, A>(
        &'scope self,
        tensors: &'tensor [A],
        feeds: &[Feed<'feed, F>],
    ) -> Vec<Option<NdArray<F>>>
        where
            A: AsRef<Tensor<'tensor, 'scope, F>> + Copy,
    {
        // Panics if given shapes are invalid
        validate_feed_shapes(feeds, self);

        let mut _subgraph_state: SubGraph<'tensor, F> = build_stateful_subgraph_from(tensors, self);
        let mut _owned_storage = ValueStorage { inner: Vec::new() };
        let mut _view_storage = ViewStorage { inner: Vec::new() };

        // prepare mut refs to avoid moves in the main loop below
        let owned_storage = &mut _owned_storage;
        let view_storage = &mut _view_storage;
        let subgraph_state = &mut _subgraph_state;

        // rayon scope
        // - blocks until all nodes in the subgraph are processed.
        // - generates the return value of this function
        crate::rayon::scope(move |rayon_scope| {
            let (tx, rx) = crossbeam_channel::unbounded();
//            let mut _lock_guard_registry = LockGuardRegistry::new();
//            let lock_guard_registry = &mut _lock_guard_registry;

            // schedule source nodes.
            for &src in &subgraph_state.sources {
                tx
                    .send(OpEvalResult {
                        tensor: src,
                        ys: if !src.requires_compute() {
                            ArrayVec::<[_; NUM_MAX_OUTPUT]>::new()
                        } else {
                            let mut ctx = OpComputeContext::new(src, vec![]);
                            src.op.compute(&mut ctx);
                            let ys = ctx.extract_outputs();
                            if ys.is_empty() {
                                panic!("Bad op implementation: empty return value");
                            }
                            ys
                        },
                    })
                    .unwrap();
            }

            // main loop.
            loop {
                // aggregate and register a compute result.
                let evaluated = {
                    let OpEvalResult { tensor, ys } = rx.recv().unwrap();

//                    lock_guard_registry.deregister_input_guards_of(tensor);

                    let node = subgraph_state.map.get_mut(&tensor.id()).unwrap();
                    if tensor.requires_compute() {
                        extract_compute_results_async(ys, owned_storage, view_storage, node);
                    }
                    // removing mutability...
                    node as *const _
                };
                let evaluated: &NodeInfoAsync<F> = unsafe { &*evaluated };

                if subgraph_state.status.maybe_update_with(evaluated) == GraphStatus::Completed {
                    // exit the main loop!
                    break;
                }

                // try to schedule the successors of the evaluated node.
                for &suc in &evaluated.successors {
                    let sender = tx.clone();
                    let suc_info = subgraph_state.get(suc);
                    suc_info.decrement_pending_count();

                    if !suc_info.scheduled() && suc_info.ready() {
                        suc_info.mark_scheduled();
//                        let suc_input_persistent_arrays = &suc_info.in_persistent_arrays;
//                        lock_guard_registry.init(suc);

                        rayon_scope.spawn(move |_| {
                            let mut xs = Vec::with_capacity(suc.in_edges.len());

                            // Aggregate in_node's inputs
//                            let mut guards_mut = Vec::new();
//                            for (i, ((input, &in_idx), in_arr)) in suc.in_edges.iter()
                            for (i, (input, &in_idx)) in suc.in_edges.iter()
                                .zip(&suc.input_indices)
//                                .zip(suc_input_persistent_arrays)
                                .enumerate() {
                                let x = if input.is_placeholder {
                                    OpInput::new(retrieve_feed(feeds, input.id))
//                                } else if let PersistentArray::Variable(ref lock) = in_arr {
//                                    if input.mut_usage {
//                                        let guard_mut = suc.lock_variable_array_mut().unwrap();
//                                        guards_mut.push(guard_mut);
//                                        OpInput::new(retrieve_feed(feeds, input.id))
////                                        OpInput::new_mut(lock_guard_registry.lock_rw(suc.id(), i, lock).view_mut())
//                                    } else {
//                                        OpInput::new(retrieve_feed(feeds, input.id))
////                                                OpInput::new(lock_guard_registry.lock_write(suc.id(), i, lock).view())
//                                    }
//                                } else if let PersistentArray::Constant(ref arr) = in_arr {
//                                    OpInput::new(arr.view())
                                } else {
                                    // Retrieve the output of other nodes
                                    let vi = &subgraph_state.map.get(&input.id).unwrap().value_info_list[in_idx];
                                    // NOTE: input views are not tracked by borrow checker but it's ok because
                                    // - Only the main thread can mutate the output storage.
                                    // - Every item in the storage is thread safe.
                                    // - Once an item is placed in the storage, that exists there until the storage dropped.
                                    match vi.ty {
                                        ValueType::Owned => unsafe {
                                            OpInput::new((owned_storage.get_untracked_view(vi.key)))
                                        },
                                        ValueType::View => unsafe {
                                            OpInput::new((view_storage.get_untracked_view(vi.key)))
                                        },
                                        ValueType::Empty => {
                                            panic!("Attempting to use an empty output as an op's input.");
                                        }
                                    }
                                };
                                xs.push(x);
                            }

                            // run compute
                            let mut ctx = OpComputeContext::new(suc, xs);
                            suc.op.compute(&mut ctx);
                            let ys = ctx.extract_outputs();
                            if ys.is_empty() {
                                panic!("Bad op implementation: empty return value");
                            }
//                            std::mem::drop(guards_mut);
                            sender.send(OpEvalResult { tensor: suc, ys }).unwrap();
                        });
                    }
                }
            }

            // aggregate return values
            let target_statuses = &subgraph_state.status.target_statuses;
            let mut ret: Vec<Option<NdArray<F>>> = Vec::with_capacity(target_statuses.len());
            for (id, status) in target_statuses {
//                debug_assert_eq!(status, NodeStatus::Completed);
                let node = subgraph_state.map.get(&id).unwrap();
                let owned_value = if let Some(per) = node.clone_persistent_array() {
                    Some(per)
                } else if node.is_placeholder {
                    Some(retrieve_feed(feeds, *id).to_owned())
                } else {
                    None
//                    let info = &node.value_info_list[0];
//                    match info.ty {
//                        ValueType::Owned => {
//                            mem::replace(&mut storage.owned_mut()[info.key], None)
//                        },
//                        ValueType::View => {
//                            Some(storage.view()[info.key].to_owned())
//                        },
//                        ValueType::Empty => {
//                            None
//                        }
//                    }
                };
                ret.push(owned_value);
            }
//            }

            ret
        }
        )
    }

    #[inline]
    fn would_not_visit<'t>(node: &TensorInternal<F>, info_map: &FxHashMap<usize, NodeInfo<'t, F>>) -> bool {
        node.is_placeholder || node.has_persistent_array || info_map.contains_key(&node.id())
    }

//    /// Evaluates given symbolic tensors.
//    ///
//    /// Each return value can be `None`;
//    /// for example, evaluation of `gradient_descent_ops::*`
//    /// would result in `None`.
//    ///
//    /// NOTE: All the runtime errors are not reported by return values, but by *panic*.
//    ///
//    /// ```
//    /// use ndarray::array;
//    /// use autograd as ag;
//    ///
//    /// ag::with(|g| {
//    ///     let a = g.zeros(&[2]);
//    ///     let b = g.ones(&[2]);
//    ///
//    ///     // eval two tensors at once.
//    ///     let evaluated = g.eval(&[a, b], &[]);
//    ///     assert_eq!(evaluated[0], Some(array![0., 0.].into_dyn()));
//    ///     assert_eq!(evaluated[1], Some(array![1., 1.].into_dyn()));
//    /// });
//    /// ```
//    /// See also [Tensor::eval](tensor/struct.Tensor.html#method.eval).
//    pub fn eval<'feed, 'tensor, 'scope: 'tensor, A>(
//        &'scope self,
//        tensors: &'tensor [A],
//        feeds: &[Feed<'feed, F>],
//    ) -> Vec<Option<NdArray<F>>>
//    where
//        A: AsRef<Tensor<'tensor, 'scope, F>> + Copy,
//    {
//        validate_feed_shapes(feeds, self);
//
//        let mut node_info_map: FxHashMap<usize, NodeInfo<'tensor, F>> = FxHashMap::default();
//
//        // Storage in which compute results are stored.
//        let mut storage = OutputStorage::new();
//
//        // Storage for RAII guards of variable locks
//        let lock_state = LockGuardRegister::new();
//
//        let mut dfs_stack = Vec::<(&TensorInternal<F>, bool)>::with_capacity(100);
//        for t in tensors.iter() {
//            dfs_stack.push((t.as_ref().tensor, false));
//        }
//
//        while let Some((node, is_parent)) = dfs_stack.pop() {
//            if is_parent {
//                if Self::would_not_visit(node, &node_info_map) {
//                    continue;
//                }
//
//                // Aggregate inputs for `in_node`
//                let mut xs = Vec::with_capacity(node.in_edges.len());
//                // TODO
//                //                lock_state.init(node);
//                for (i, (in_node, &in_idx)) in
//                    node.in_edges.iter().zip(&node.input_indices).enumerate()
//                {
//                    let input_inner = in_node.get(self).tensor;
//                    let x = if input_inner.is_placeholder {
//                        OpInput::new(retrieve_feed(feeds, in_node.id))
//                    } else if let Some(ref lock) = input_inner.variable_array {
//                        // TODO (以下を消して、コメントアウトを解除)
//                        OpInput::new(retrieve_feed(feeds, in_node.id))
//                    // TODO
//                    //                        if input.mut_usage {
//                    //                            OpInput::new_mut(lock_state.lock_rw(node.id(), i, lock).view_mut())
//                    //                        } else {
//                    //                            OpInput::new(lock_state.lock_write(node.id(), i, lock).view())
//                    //                        }
//                    } else if let Some(ref arr) = input_inner.get_constant_array() {
//                        OpInput::new(arr.view())
//                    } else {
//                        // Search the output of other nodes
//                        let vi = &node_info_map.get(&in_node.id).unwrap().value_info_list[in_idx];
//                        match vi.ty {
//                            ValueType::Owned => {
//                                OpInput::new(storage.owned()[vi.key].as_ref().unwrap().view())
//                            }
//                            ValueType::View => OpInput::new(storage.view()[vi.key].clone()),
//                            ValueType::Empty => {
//                                panic!(
//                                    "Attempting to use {}'s output which is empty.",
//                                    input_inner.op.name()
//                                );
//                            }
//                        }
//                    };
//                    xs.push(x);
//                }
//
//                // run compute
//                let mut ctx = OpComputeContext::new(node, xs);
//                node.op.compute(&mut ctx);
//                // TODO
//                //                lock_state.invalidate_input_guards_of(node);
//                let ys = ctx.extract_outputs();
//                if ys.is_empty() {
//                    panic!("Bad op implementation: empty return value");
//                }
//                // register compute result
//                let node_info = extract_compute_results(ys, &mut storage, node);
//                node_info_map.insert(node.id(), node_info);
//            } else {
//                // Update dfs stack
//                dfs_stack.push((node, true));
//                // Push children if needed
//                for child in &node.in_edges {
//                    if !Self::would_not_visit(child.get(self).tensor, &node_info_map) {
//                        dfs_stack.push((child.get(self).tensor, false));
//                    }
//                }
//            }
//        }
//
//        // Aggregate return values
//        let mut ret = Vec::with_capacity(tensors.len());
//        for t in tensors {
//            let t = t.as_ref();
//            let arr = if let Some(per) = t.clone_persistent_array() {
//                Some(per)
//            } else if t.is_placeholder() {
//                Some(retrieve_feed(feeds, t.id()).to_owned())
//            } else {
//                let info = &node_info_map.get(&t.id()).unwrap().value_info_list[0];
//                if ValueType::Owned == info.ty {
//                    mem::replace(&mut storage.owned_mut()[info.key], None)
//                } else if ValueType::View == info.ty {
//                    Some(storage.view()[info.key].to_owned())
//                } else {
//                    None
//                }
//            };
//            ret.push(arr);
//        }
//        ret
//    }
}

#[test]
fn test_eval() {
    crate::with(|g| {
        let v: Tensor<f32> = g.placeholder(&[3, 2, 1]);
        let z = g.reduce_sum(g.squeeze(v, &[2]), &[0, 1], false);
        let g = g.grad(&[z], &[v]);
        let eval_result = g[0].eval(&[v.given(crate::ndarray_ext::ones(&[3, 2, 1]).view())]);
        assert_eq!(eval_result.as_ref().unwrap().shape(), &[3, 2, 1]);
    })
}

#[test]
fn test_constant_eval() {
    use crate::tensor::Constant;
    crate::with(|g| {
        let arr = ndarray::arr1(&[0., 0., 0.]).into_dyn();
        assert_eq!(Some(arr.clone()), g.constant(arr).eval(&[]));
    });
}

#[test]
fn test_placeholder_eval() {
    crate::with(|g| {
        let arr: NdArray<f32> = crate::ndarray_ext::ones(&[3, 2, 1]);
        let v = g.placeholder(&[3, 2, 1]);
        let eval_result = v.eval(&[v.given(arr.view())]);
        assert_eq!(eval_result, Some(arr));
    });
}
