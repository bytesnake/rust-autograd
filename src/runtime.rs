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
    pending_count: usize,
    scheduled: bool,
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
            pending_count: 0,
            scheduled: false,
        }
    }

    #[inline]
    fn scheduled(&self) -> bool {
        self.scheduled
    }

    #[inline]
    fn mark_scheduled(&mut self) {
//        self.scheduled.set(true);
        self.scheduled = true;
    }

    #[inline]
    fn increment_pending_count(&mut self) {
        self.pending_count += 1;
//        self.pending_count.set(self.pending_count.get() + 1);
    }

    #[inline]
    fn decrement_pending_count(&mut self) {
        self.pending_count = self.pending_count.saturating_sub(1);
//        self.pending_count
//            .set(self.pending_count.get().saturating_sub(1));
    }

    #[inline]
    fn ready(&self) -> bool {
//        self.pending_count.get() == 0
        self.pending_count == 0
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
    compute_was_called: bool,
    ys: crate::op::ComputeResults<'v, T>,
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

type Receiver<'t, 'v, T: Float> = crossbeam_channel::Receiver<OpEvalResult<'t, 'v, T>>;
type Sender<'t, 'v, T: Float> = crossbeam_channel::Sender<OpEvalResult<'t, 'v, T>>;

// Builds a subgraph consisting of nodes that are reachable from `tensors`.
// Returns: graph, source_nodes
fn build_minimum_subgraph_from<K, T>(targets:&[K]) -> SubGraph<T>
    where
        K:AsRef<Tensor<T>>,
        T:Float,
{
    // mapping of tensor -> node-info
    let mut graph = FxHashMap::<&Tensor<T>, Node<T>>::default();
    // set of source nodes
    let mut sources = FxHashSet::default();

    let mut dfs_stack: Vec<&Tensor<_>> = Vec::with_capacity(128);

    // Initialize the graph and stack with `targets`
    for t in targets {
        let t = t.as_ref();
        let node = Node::new(t, None);
        if let Entry::Vacant(ent) = graph.entry(t) {
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
            let mut inc = true;
            match graph.entry(&child.val) {
                Entry::Vacant(ent) => { // initial visit
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
                        inc = false;
                    }
                }
            }
            if inc {
                graph.get_mut(&node).unwrap().increment_pending_count();
            }
        }
    }
    SubGraph {
        sources,
        node_map: graph
    }
}

struct SubGraph<'a, F: Float> {
    sources: FxHashSet<&'a Tensor<F>>,
    node_map: FxHashMap<&'a Tensor<F>, Node<'a, F>>
}

struct Evaluator<'tensor, 'view, F: Float, T: AsRef<Tensor<F>>> {
    targets: &'tensor [T],
    graph: SubGraph<'tensor, F>,
    sender: Sender<'tensor, 'view, F>,
    receiver: Receiver<'tensor, 'view, F>,
}

struct EvaluatorStorage<'tensor, 'view, F: Float> {
    value_info_storage: FxHashMap<&'tensor Tensor<F>, Vec<ValueInfo>>,
    owned_value_storage: Vec<Option<NdArray<F>>>,
    borrowed_value_storage: Vec<NdArrayView<'view, F>>,
}

struct GuardState<'tensor, 'lock, F: Float> {
    read_guards_map: UnsafeCell<FxHashMap<&'tensor Tensor<F>, Vec<Option<RwLockReadGuard<'lock, NdArray<F>>>>>>,
    write_guards_map: UnsafeCell<FxHashMap<&'tensor Tensor<F>, Vec<Option<RwLockWriteGuard<'lock, NdArray<F>>>>>>,
}

impl<'tensor, 'view, 'lock, F: Float> GuardState<'tensor, 'lock, F> {
    fn new() -> Self {
        GuardState {
            read_guards_map: UnsafeCell::new(FxHashMap::default()),
            write_guards_map: UnsafeCell::new(FxHashMap::default()),
        }
    }
}

impl<'tensor, 'view, 'lock, F: Float> EvaluatorStorage<'tensor, 'view, F> {
    fn new() -> Self {
        EvaluatorStorage {
            value_info_storage: FxHashMap::default(),
            owned_value_storage: Vec::new(),
            borrowed_value_storage: Vec::new(),
        }
    }
}

impl<'tensor, 'view, 'lock, F: Float, T: Sync + AsRef<Tensor<F>>> Evaluator<'tensor, 'view, F, T> {

    #[inline]
    pub fn new(targets: &'tensor [T]) -> Self {
        let (sender, receiver) = crossbeam_channel::bounded(16);
        Evaluator {
            targets,
            graph: build_minimum_subgraph_from(targets),
            sender,
            receiver,
        }
    }

    fn send_sources(&self) {
        for src in self.graph.sources {
            // source: placeholder, constant, variable, generator.
            let is_basic = src.is_placeholder || src.has_persistent_array;
            self.sender.send(OpEvalResult {
                node: src,
                compute_was_called: !is_basic,
                ys: if is_basic {
                    vec![]
                } else {
                    let mut ctx = OpComputeContext::new(src, vec![]);
                    src.op.compute(&mut ctx);
                    ctx.ys.unwrap()
                },
            });
        }
    }

    pub fn eval<'slice, 'node, 'feed>(
        &mut self,
        tensors: &'node [T],
        feeds: &'slice [Feed<'node, 'feed, F>],
    ) -> Vec<Option<NdArray<F>>>
        where
            T: AsRef<Tensor<F>>,
            F: Float,
    {
        let mut storage = EvaluatorStorage::new();
        let mut guards = GuardState::new();

        self.send_sources();

        let mut targets_remaining = tensors.len();

        loop {
            // Aggregate and register a compute result.
            let evaluated: &Node<F> = unsafe {
                self.recv(&mut storage, &mut guards)
            };

            if is_eval_target(evaluated.inner, tensors) {
                targets_remaining -= 1;
                if targets_remaining == 0 {
                    break; // exit the main loop as all the target tensors were evaluated.
                }
            }

            // Try to schedule the immediate successors of the evaluated node.
            for &suc in &evaluated.successors {
                let mut suc_info = self.graph.node_map.get(&suc).unwrap();
                suc_info.decrement_pending_count();

                if !suc_info.scheduled() && suc_info.ready() {
                    suc_info.mark_scheduled();
                    // Prepare and send inputs to executor.
                    unsafe {
                        (&mut *guards.write_guards_map.get()).insert(suc, crate::none_vec(suc.inputs.len()));
                        (&mut *guards.read_guards_map.get()).insert(suc, crate::none_vec(suc.inputs.len()));
                    }

                    let mut xs = Vec::with_capacity(suc.inputs.len());

                    // Aggregate inputs for `in_node`
                    for (i, (in_node, &in_idx)) in suc.inputs.iter().zip(&suc.input_indices).enumerate()
                        {
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
                                        let mut got_mut: &mut Vec<_> =
                                            (&mut *guards.write_guards_map.get()).get_mut(&suc).unwrap();
                                        got_mut[i] = Some(lock.write().unwrap());
                                        xs.push(OpInput::new_mut(got_mut[i].as_mut().unwrap().view_mut()));
                                    } else {
                                        let mut got_mut: &mut Vec<_> =
                                            (&mut *guards.read_guards_map.get()).get_mut(&suc).unwrap();
                                        got_mut[i] = Some(lock.read().unwrap());
                                        xs.push(OpInput::new(got_mut[i].as_ref().unwrap().view()));
                                    }
                                }
                            } else if let Some(ref arr) = in_node.get_constant_array() {
                                xs.push(OpInput::new(arr.view()));
                            } else {
                                // Search the output of other nodes
                                let info = &storage.value_info_storage.get(&&in_node.val).unwrap()[in_idx];
                                match info.ty {
                                    ValueType::Owned => unsafe {
                                        xs.push(OpInput::new(storage.owned_value_storage[info.key].as_ref().unwrap().view()));
                                    },
                                    ValueType::View => {
                                        // Clone the view
                                        xs.push(OpInput::new(storage.borrowed_value_storage[info.key].clone()));
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
                            self.sender.send(OpEvalResult {
                                node: suc,
                                compute_was_called: true,
                                ys: ctx.ys.expect("Bad op implementation: empty return value"),
                            });
                        })
                    });
                }
            }
        }

        let owned_storage = &mut storage.owned_value_storage;

        let mut ret: Vec<Option<NdArray<F>>> = Vec::with_capacity(tensors.len());
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
                let info = &storage.value_info_storage.get(&t).unwrap()[0];
                if ValueType::Owned == info.ty {
                    mem::replace(&mut owned_storage[info.key], None)
                } else if ValueType::View == info.ty {
                    Some(storage.borrowed_value_storage[info.key].to_owned())
                } else {
                    None
                }
            };
            ret.push(arr);
        }
        ret
    }

    unsafe fn recv<'a>(
        &'a mut self,
        storage: &mut EvaluatorStorage<'tensor, 'view, F>,
        guards: &mut GuardState<'tensor, 'lock, F>
    ) -> &'a Node<F> {
        let OpEvalResult {
            node,
            compute_was_called,
            ys,
        } = self.receiver.recv().unwrap();

        for (i, input) in self.graph.node_map
            .get(&node)
            .unwrap()
            .inner
            .inputs
            .iter()
            .enumerate() {
                if input.mut_usage {
                    mem::swap(
                        &mut (&mut *guards.write_guards_map.get()).get_mut(&node).unwrap()[i],
                        &mut None,
                    );
                } else {
                    mem::swap(
                        &mut (&mut *guards.read_guards_map.get()).get_mut(&node).unwrap()[i],
                        &mut None,
                    );
                }
        }
        if compute_was_called {
            let mut info_list = Vec::with_capacity(ys.len());
            for y in ys {
                match y {
                    Ok(crate::ArrRepr::Owned(val)) => {
                        info_list.push(ValueInfo {
                            ty: ValueType::Owned,
                            key: storage.owned_value_storage.len(),
                        });
                        storage.owned_value_storage.push(Some(val));
                    }
                    Ok(crate::ArrRepr::View(val)) => {
                        info_list.push(ValueInfo {
                            ty: ValueType::View,
                            key: storage.borrowed_value_storage.len(),
                        });
                        storage.borrowed_value_storage.push(val);
                    }
                    _ => {
                        info_list.push(ValueInfo {
                            ty: ValueType::Empty,
                            key: 0,
                        });
                    }
                }
            }
            storage.value_info_storage.insert(node, info_list);
        }
        self.graph.node_map.get(&node).unwrap()
    }
}

pub fn eval<'slice, 'node, 'feed, T, F>(
    tensors: &'node [T],
    feeds: &'slice [Feed<'node, 'feed, F>],
) -> Vec<Option<NdArray<F>>>
    where
        T: AsRef<Tensor<F>>,
        F: Float,
{
    let graph = build_minimum_subgraph_from(tensors);

    let mut output_info_storage = FxHashMap::<&Tensor<F>, Vec<ValueInfo>>::default();

    // Storage in which compute results are stored.
    let mut owned_storage = UnsafeCell::new(Vec::<Option<NdArray<F>>>::with_capacity(100));
    let mut read_guards_map =
        UnsafeCell::new(FxHashMap::<&Tensor<F>, Vec<Option<RwLockReadGuard<NdArray<F>>>>>::default());
    let mut write_guards_map =
        UnsafeCell::new(FxHashMap::<&Tensor<F>, Vec<Option<RwLockWriteGuard<_>>>>::default());

    // Views of owned arrays whose lifetime is shorter than owned ones
    let mut view_storage = Vec::<NdArrayView<F>>::new();

    let (sender, receiver) = crossbeam_channel::bounded(16);

    for src in graph.sources {
        // source: placeholder, constant, variable, generator.
        let is_basic = src.is_placeholder || src.has_persistent_array;
        sender.send(OpEvalResult {
            node: src,
            compute_was_called: !is_basic,
            ys: if is_basic {
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
        let evaluated: &Node<F> = unsafe {
            let OpEvalResult {
                node,
                compute_was_called,
                ys,
            } = receiver.recv().unwrap();
            for (i, input) in graph.node_map
                .get(&node)
                .unwrap()
                .inner
                .inputs
                .iter()
                .enumerate()
                {
                    if input.mut_usage {
                        mem::swap(
                            &mut (&mut *write_guards_map.get()).get_mut(&node).unwrap()[i],
                            &mut None,
                        );
                    } else {
                        mem::swap(
                            &mut (&mut *read_guards_map.get()).get_mut(&node).unwrap()[i],
                            &mut None,
                        );
                    }
                }
            if compute_was_called {
                let mut info_list = Vec::with_capacity(ys.len());
                for y in ys {
                    match y {
                        Ok(crate::ArrRepr::Owned(val)) => {
                            info_list.push(ValueInfo {
                                ty: ValueType::Owned,
                                key: (&*owned_storage.get()).len(),
                            });
                            (&mut *owned_storage.get()).push(Some(val));
                        }
                        Ok(crate::ArrRepr::View(val)) => {
                            info_list.push(ValueInfo {
                                ty: ValueType::View,
                                key: view_storage.len(),
                            });
                            view_storage.push(val);
                        }
                        _ => {
                            info_list.push(ValueInfo {
                                ty: ValueType::Empty,
                                key: 0,
                            });
                        }
                    }
                }
                output_info_storage.insert(node, info_list);
            }
            graph.node_map.get(&node).unwrap()
        };

        if is_eval_target(evaluated.inner, tensors) {
            targets_remaining -= 1;
            if targets_remaining == 0 {
                break; // exit the main loop as all the target tensors were evaluated.
            }
        }

        // Try to schedule the immediate successors of the evaluated node.
        for &suc in &evaluated.successors {
            let suc_info = graph.node_map.get(&suc).unwrap();
//            suc_info.decrement_pending_count();

            if !suc_info.scheduled() && suc_info.ready() {
                suc_info.mark_scheduled();
                // Prepare and send inputs to executor.
                unsafe {
                    (&mut *write_guards_map.get()).insert(suc, crate::none_vec(suc.inputs.len()));
                    (&mut *read_guards_map.get()).insert(suc, crate::none_vec(suc.inputs.len()));
                }

                let mut xs = Vec::with_capacity(suc.inputs.len());

                // Aggregate inputs for `in_node`
                for (i, (in_node, &in_idx)) in suc.inputs.iter().zip(&suc.input_indices).enumerate()
                    {
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
                                    let mut got_mut: &mut Vec<_> =
                                        (&mut *write_guards_map.get()).get_mut(&suc).unwrap();
                                    got_mut[i] = Some(lock.write().unwrap());
                                    xs.push(OpInput::new_mut(got_mut[i].as_mut().unwrap().view_mut()));
                                } else {
                                    let mut got_mut: &mut Vec<_> =
                                        (&mut *read_guards_map.get()).get_mut(&suc).unwrap();
                                    got_mut[i] = Some(lock.read().unwrap());
                                    xs.push(OpInput::new(got_mut[i].as_ref().unwrap().view()));
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
                                    // Clone the view
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
                            compute_was_called: true,
                            ys: ctx.ys.expect("Bad op implementation: empty return value"),
                        });
                    })
                });
            }
        }
    }

    let owned_storage = unsafe { &mut *owned_storage.get() };

    let mut ret: Vec<Option<NdArray<F>>> = Vec::with_capacity(tensors.len());
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
