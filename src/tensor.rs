//! Defining things related to `ag::Tensor`.
use crate::op;
use crate::Float;
use crate::NdArray;

use crate::graph::Graph;
use crate::op::OpGradientContext;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::{Add, Div, Mul, Sub};
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

/// Symbolic multi-dimensional array.
///
/// `Tensor` is basically created by operations of a `Graph`.
/// It is cheap to copy since it contains only refs to the owned internal objects.
#[derive(Clone, Copy)]
pub struct Tensor<'tensor, 'graph, F: Float> {
    pub(crate) tensor: &'tensor TensorInternal<F>,
    pub graph: &'graph Graph<F>,
}

impl<'graph, 'tensor, F: Float> Tensor<'tensor, 'graph, F> {
    /// Evaluates this tensor as an ndarray object.
    ///
    /// ```
    /// use ndarray::array;
    /// use autograd as ag;
    ///
    /// ag::with(|g| {
    ///    let a = g.zeros(&[2]);
    ///    assert_eq!(a.eval(&[]), Some(array![0., 0.].into_dyn()));
    /// });
    /// ```
    ///
    /// See also [Graph::eval](../graph/struct.Graph.html#method.eval).
    pub fn eval<'v>(&'tensor self, feeds: &'v [crate::runtime::Feed<'v, F>]) -> Option<NdArray<F>> {
        let mut ret = self.graph.eval(&[self], feeds);
        debug_assert_eq!(ret.len(), 1);
        ret.remove(0)
    }

    /// Retruns a `Feed` assigning a given value to this (placeholder) tensor.
    ///
    /// Ensure that the return value is passed to `ag::Eval`, `ag::eval` or `Tensor::eval`.
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
    ///     x.eval(&[
    ///         x.given(value.view())
    ///     ]);
    /// });
    /// ```
    pub fn given<D>(self, value: ndarray::ArrayView<F, D>) -> crate::runtime::Feed<F>
    where
        D: ndarray::Dimension,
    {
        assert!(
            self.is_placeholder(),
            "Callee of Tensor::given must be a placeholder."
        );
        crate::runtime::Feed::new(self.id(), value.into_dyn())
    }

    #[inline]
    /// Creates a new `TensorBuilder`.
    pub fn builder() -> TensorBuilder<F> {
        TensorBuilder {
            shape: None,
            inputs: Vec::new(),
            can_have_gradient: true,
            constant_array: None,
            variable_array: None,
            is_placeholder: false,
            input_indices: None,
            backprop_inputs: None,
            known_shape: None,
        }
    }

    /// Registers a hook on a `Tensor`.
    ///
    /// Pre-defined hooks are descripted [here](../hook/index.html).
    ///
    /// ```
    /// use autograd as ag;
    ///
    /// ag::with(|g| {
    ///     let a: ag::Tensor<f32> = g.zeros(&[4, 2]).hook(ag::hook::Show);
    ///     let b: ag::Tensor<f32> = g.ones(&[2, 3]).hook(ag::hook::ShowShape);
    ///     let c = g.matmul(a, b);
    ///
    ///     c.eval(&[]);
    ///     // [[0.0, 0.0],
    ///     // [0.0, 0.0],
    ///     // [0.0, 0.0],
    ///     // [0.0, 0.0]] shape=[4, 2], strides=[2, 1], layout=C (0x1)
    ///
    ///     // [2, 3]
    /// });
    /// ```
    #[inline]
    pub fn hook<H: crate::hook::Hook<F> + Sync + 'static>(
        self,
        hook: H,
    ) -> Tensor<'tensor, 'graph, F> {
        Tensor::builder()
            .set_input(&self)
            .build(self.graph, crate::ops::hook_ops::HookOp::new(hook))
    }

    /// Shorthand for `Tensor::hook(ag::hook::Show)`
    ///
    /// See also [hook::Show](../hook/struct.Show.html).
    ///
    /// ```
    /// use autograd as ag;
    ///
    /// ag::with(|g| {
    ///     let a: ag::Tensor<f32> = g.zeros(&[4, 2]).show();
    ///     a.eval(&[]);
    ///     // [[0.0, 0.0],
    ///     // [0.0, 0.0],
    ///     // [0.0, 0.0],
    ///     // [0.0, 0.0]] shape=[4, 2], strides=[2, 1], layout=C (0x1)
    ///     });
    /// ```
    #[inline]
    pub fn show(self) -> Tensor<'tensor, 'graph, F> {
        self.hook(crate::hook::Show)
    }

    /// Shorthand for `Tensor::hook(ag::hook::Show)`
    ///
    /// See also [hook::ShowWith](../hook/struct.ShowWith.html).
    ///
    /// ```
    /// use autograd as ag;
    ///
    /// ag::with(|g| {
    ///     let a: ag::Tensor<f32> = g.zeros(&[4, 2]).show_with("My value:");
    ///     a.eval(&[]);
    ///     // My value:
    ///     // [[0.0, 0.0],
    ///     // [0.0, 0.0],
    ///     // [0.0, 0.0],
    ///     // [0.0, 0.0]] shape=[4, 2], strides=[2, 1], layout=C (0x1)
    /// });
    ///
    /// ```
    #[inline]
    pub fn show_with(self, what: &'static str) -> Tensor<'tensor, 'graph, F> {
        self.hook(crate::hook::ShowWith(what))
    }

    /// Shorthand for `Tensor::hook(ag::hook::ShowShape)`
    ///
    /// See also [hook::ShowShape](../hook/struct.ShowShape.html).
    ///
    /// ```
    /// use autograd as ag;
    ///
    /// ag::with(|g| {
    ///     let a: ag::Tensor<f32> = g.zeros(&[2, 3]).show_shape();
    ///     a.eval(&[]);
    ///     // [2, 3]
    /// });
    /// ```
    #[inline]
    pub fn show_shape(self) -> Tensor<'tensor, 'graph, F> {
        self.hook(crate::hook::ShowShape)
    }

    /// Shorthand for `Tensor::hook(ag::hook::ShowShape)`
    ///
    /// See also [hook::ShowShapeWith](../hook/struct.ShowShapeWith.html).
    ///
    /// ```
    /// use autograd as ag;
    ///
    /// ag::with(|g| {
    ///     let a: ag::Tensor<f32> = g.zeros(&[2, 3]).show_shape_with("My shape:");
    ///     a.eval(&[]);
    ///     // My shape:
    ///     // [2, 3]
    /// });
    /// ```
    #[inline]
    pub fn show_shape_with(self, what: &'static str) -> Tensor<'tensor, 'graph, F> {
        self.hook(crate::hook::ShowShapeWith(what))
    }

    /// Shorthand for `Tensor::hook(ag::hook::PrintAny("what"))`
    ///
    /// See also [hook::PrintAny](../hook/struct.PrintAny.html).
    ///
    /// ```
    /// use autograd as ag;
    ///
    /// ag::with(|g| {
    ///     let a: ag::Tensor<f32> = g.zeros(&[2, 3]).print_any("This is `a`");
    ///     a.eval(&[]);
    ///     // This is `a`
    /// });
    /// ```
    #[inline]
    pub fn print_any(self, what: &'static str) -> Tensor<'tensor, 'graph, F> {
        self.hook(crate::hook::PrintAny(what))
    }

    /// Returns the id of this tensor.
    #[inline]
    pub fn id(&self) -> usize {
        self.tensor.id()
    }

    #[inline]
    /// Returns true if this node has no incoming nodes.
    pub fn is_source(&self) -> bool {
        self.tensor.in_edges.is_empty()
    }

    #[inline]
    /// Returns
    pub fn get_backprop_inputs(&self) -> &[Input] {
        self.tensor.get_backprop_inputs()
    }

    #[inline]
    pub fn is_placeholder(&self) -> bool {
        self.tensor.is_placeholder
    }

    #[inline]
    pub fn clone_persistent_array(&self) -> Option<NdArray<F>> {
        self.tensor.clone_persistent_array()
    }

    #[inline]
    pub fn get_constant_array(&self) -> Option<&NdArray<F>> {
        self.tensor.get_constant_array()
    }

    #[inline]
    pub fn get_variable_array(&self) -> Option<&RwLock<NdArray<F>>> {
        self.tensor.get_variable_array()
    }

    #[inline]
    pub fn lock_variable_array(&self) -> Option<RwLockReadGuard<NdArray<F>>> {
        self.tensor.lock_variable_array()
    }

    #[inline]
    pub fn lock_variable_array_mut(&self) -> Option<RwLockWriteGuard<NdArray<F>>> {
        self.tensor.lock_variable_array_mut()
    }

    #[inline]
    pub fn is_differentiable(&self) -> bool {
        self.tensor.is_differentiable
    }

    /// True is this tensor was created by `Graph::variable`.
    #[inline]
    #[allow(dead_code)]
    pub(crate) fn is_variable(&self) -> bool {
        self.tensor.variable_array.is_some()
    }
}

impl<'a, 'b, T: Float> AsRef<Tensor<'a, 'b, T>> for Tensor<'a, 'b, T> {
    #[inline(always)]
    fn as_ref(&self) -> &Tensor<'a, 'b, T> {
        self
    }
}

pub(crate) struct TensorInternal<T: Float> {
    pub(crate) id: usize,

    // Operation to evaluate this tensor.
    pub(crate) op: Box<dyn op::Op<T> + Sync>,

    // References to immediate predecessors.
    pub(crate) in_edges: Vec<Input>,

    // The rank number for topological ordering in a graph.
    pub(crate) top_rank: usize,

    // *Symbolic* shape of this tensor.
    pub(crate) shape: Option<usize>,

    // An optional *persistent* NdArray.
    //
    // This is `Some` if this tensor is made from `ag::variable`.
    pub(crate) variable_array: Option<Arc<RwLock<NdArray<T>>>>,

    // An optional *persistent* NdArray.
    //
    // This is `Some` if this tensor is made from `ag::constant`.
    pub(crate) constant_array: Option<Arc<NdArray<T>>>,

    // This tensor is placeholder or not.
    pub(crate) is_placeholder: bool,

    // This is `True` if this tensor can have gradient for any objectives.
    pub(crate) is_differentiable: bool,

    // This is `Some` if this tensor is made from `ag::constant` or `ag::variable`.
    pub(crate) has_persistent_array: bool,

    /// Input indices of arrays used in `compute`
    pub(crate) input_indices: Vec<usize>,

    /// Input nodes used when backprop.
    ///
    /// This is same as `inputs` in most cases.
    pub(crate) backprop_inputs: Option<Vec<Input>>,

    /// Static shape of this tensor.
    /// Each dim size is *signed* for placeholders.
    pub(crate) known_shape: Option<KnownShape>,
}

pub(crate) enum PersistentArray<'t, F: Float> {
    Variable(&'t RwLock<NdArray<F>>),
    Constant(&'t NdArray<F>),
    None
}

impl<'t, F: Float> PersistentArray<'t, F> {
    pub(crate) fn owned(&self) -> Option<NdArray<F>> {
        match self {
            PersistentArray::Variable(inner) => {
                Some(inner.read().unwrap().clone())
            },
            PersistentArray::Constant(inner) => {
                Some((*inner).clone())
            },
            PersistentArray::None => {
                None
            }
        }
    }
}


impl<T: Float> TensorInternal<T> {
    #[inline(always)]
    pub fn id(&self) -> usize {
        self.id
    }

    pub(crate) fn get_persistent_array(&self) -> PersistentArray<T> {
        if let Some(c) = self.get_constant_array() {
            PersistentArray::Constant(c)
        } else if let Some(c) = self.get_variable_array() {
            PersistentArray::Variable(c)
        } else {
            PersistentArray::None
        }
    }

    /// Returns a reference to the persistent constant array.
    ///
    /// Note that this is `Some` if this tensor derived from `ag::constant`; otherwise `None`
    #[inline]
    pub(crate) fn get_variable_array(&self) -> Option<&RwLock<NdArray<T>>> {
        match &self.variable_array {
            Some(ref inner) => Some(&**inner),
            None => None,
        }
    }

    /// Returns a reference to the persistent constant array.
    ///
    /// Note that this is `Some` if this tensor derived from `ag::constant`; otherwise `None`
    #[inline]
    pub(crate) fn get_constant_array(&self) -> Option<&NdArray<T>> {
        match &self.constant_array {
            Some(ref inner) => Some(&**inner),
            None => None,
        }
    }

    /// Locks the persistent variable tensor and returns the handle.
    ///
    /// Note that this is `Some` if this tensor derived from `ag::variable`; otherwise `None`.
    #[inline]
    pub(crate) fn lock_variable_array(&self) -> Option<RwLockReadGuard<NdArray<T>>> {
        if let Some(ref arr) = self.variable_array {
            Some(arr.read().unwrap())
        } else {
            None
        }
    }

    /// Returns a mutable reference to the persistent array.
    ///
    /// Note that this is `Some` if this tensor derived from `ag::variable`; otherwise `None`
    #[inline]
    pub(crate) fn lock_variable_array_mut(&self) -> Option<RwLockWriteGuard<NdArray<T>>> {
        if let Some(ref arr) = self.variable_array {
            Some(arr.write().unwrap())
        } else {
            None
        }
    }

    /// Returns a cloned persistent array.
    #[inline]
    pub(crate) fn clone_persistent_array(&self) -> Option<NdArray<T>> {
        if let Some(ref arr) = self.variable_array {
            Some((*arr.read().unwrap()).clone())
        } else {
            if let Some(ref arr) = self.constant_array {
                Some((**arr).clone())
            } else {
                None
            }
        }
    }

    /// True if the op of this tensor is differentiable
    #[inline]
    #[allow(dead_code)]
    pub fn is_differentiable(&self) -> bool {
        self.is_differentiable
    }

    #[inline]
    pub(crate) fn validate_feed_shape(&self, shape: &[usize]) {
        if !self.known_shape.as_ref().unwrap().validate(shape) {
            panic!(
                "Shape error: placeholder required {:?}, but got {:?}",
                self.known_shape.as_ref().unwrap().get(),
                shape
            );
        }
    }

    #[inline]
    /// Returns true if this node has no incoming nodes.
    pub(crate) fn requires_compute(&self) -> bool {
        !self.is_placeholder && !self.has_persistent_array
    }

    #[inline]
    /// Returns true if this node has no incoming nodes.
    pub(crate) fn is_source(&self) -> bool {
        self.in_edges.is_empty()
    }

    #[inline]
    /// Input nodes used when backprop.
    ///
    /// This is same as `inputs` in most cases.
    pub(crate) fn get_backprop_inputs(&self) -> &[Input] {
        self.backprop_inputs
            .as_ref()
            .unwrap_or(&self.in_edges)
            .as_slice()
    }

    #[inline]
    pub(crate) fn get_scoped_input<'a, 'b: 'a>(&self, s: &'b Graph<T>) -> Vec<Tensor<'a, 'b, T>> {
        let len = self.in_edges.len();
        let mut ret = Vec::with_capacity(len);
        for a in self.in_edges.iter() {
            ret.push(a.get(s));
        }
        ret
    }
}

impl<T: Float> fmt::Debug for TensorInternal<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "[name: {}, num of inputs: {}]",
            self.op.name(),
            self.in_edges.len()
        )
    }
}

// empty implementation
impl<T: Float> Eq for TensorInternal<T> {}

impl<T: Float> PartialEq for TensorInternal<T> {
    fn eq(&self, other: &TensorInternal<T>) -> bool {
        // compare addresses on the heap
        self.id() == other.id()
    }
}

/// Raw pointer hashing
impl<T: Float> Hash for TensorInternal<T> {
    #[inline(always)]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state)
    }
}

impl<T: Float> AsRef<TensorInternal<T>> for TensorInternal<T> {
    #[inline(always)]
    fn as_ref(&self) -> &TensorInternal<T> {
        self
    }
}

impl<T: Float> fmt::Display for TensorInternal<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "name={}", self.op.name(),)
    }
}

/// An input to a `Tensor`, which is actually a decorated tensor id.
///
/// Each of `Tensor` holds its inputs as `Vec<Input>`.
#[derive(Clone)]
pub struct Input {
    pub(crate) id: usize,
    pub(crate) mut_usage: bool,
    pub(crate) is_placeholder: bool
}

impl<'tensor, 'graph> Input {
    /// Instantiates a new immutable `Input` object.
    #[inline]
    pub fn new<T: Float>(val: &Tensor<'tensor, 'graph, T>) -> Input {
        Input {
            id: val.id(),
            mut_usage: false,
            is_placeholder: val.is_placeholder()
        }
    }

    /// Instantiates a new mutable `Input` object.
    ///
    /// Mutable sign denotes that optimizers can modify the tensor.
    #[inline]
    pub fn new_mut<T: Float>(val: &Tensor<'tensor, 'graph, T>) -> Input {
        Input {
            id: val.id(),
            mut_usage: true,
            is_placeholder: val.is_placeholder()
        }
    }

    #[inline]
    pub(crate) fn get_inner<'a, 'b: 'a, T: Float>(&self, graph: &'b Graph<T>) -> &'a TensorInternal<T> {
        graph.access_node(self.id)
    }

    #[inline]
    pub(crate) fn get<'a, 'b: 'a, T: Float>(&self, graph: &'b Graph<T>) -> Tensor<'a, 'b, T> {
        Tensor {
            tensor: graph.access_node(self.id),
            graph,
        }
    }
}

/// Builder for `ag::Tensor`
pub struct TensorBuilder<T: Float> {
    shape: Option<usize>,
    inputs: Vec<Input>,
    can_have_gradient: bool,
    is_placeholder: bool,
    constant_array: Option<Arc<NdArray<T>>>,
    variable_array: Option<Arc<RwLock<NdArray<T>>>>,
    input_indices: Option<Vec<usize>>,
    backprop_inputs: Option<Vec<Input>>,
    known_shape: Option<KnownShape>,
}

pub(crate) struct KnownShape {
    shape: Vec<isize>,
    #[allow(dead_code)]
    is_fully_defined: bool,
}

impl KnownShape {
    #[inline]
    pub(crate) fn new(shape: Vec<isize>) -> Self {
        let mut is_fully_defined = true;
        for &a in &shape {
            if a == -1 {
                is_fully_defined = false;
            } else if a <= -1 || a == 0 {
                panic!("Given shape ({:?}) contains invalid dim size(s)", &shape);
            }
        }
        Self {
            shape,
            is_fully_defined,
        }
    }

    #[inline]
    pub fn get(&self) -> &[isize] {
        self.shape.as_slice()
    }

    pub fn validate(&self, target: &[usize]) -> bool {
        if self.shape.len() != target.len() {
            return false;
        }
        for (&i, &u) in self.shape.iter().zip(target) {
            if i > 0 && i as usize != u {
                return false;
            }
        }
        true
    }

    #[inline]
    #[allow(dead_code)]
    pub fn is_fully_defined(&self) -> bool {
        self.is_fully_defined
    }
}

#[test]
fn test_build() {
    crate::with(|s| {
        let a: Tensor<f32> = s.zeros(&[4, 2]);
        let v: Tensor<f32> = s.zeros(&[2, 3]);
        let b: Tensor<f32> = s.zeros(&[4, 3]);
        let z = s.matmul(a, v) + b;
        let mut vars = [a.tensor, v.tensor, b.tensor, z.tensor];
        // `sort_by_key` don't reverse the order of `a` and `v`
        vars.sort_by_key(|a| a.top_rank);
        assert_eq!(vars, [a.tensor, v.tensor, b.tensor, z.tensor])
    });
}

impl<'tensor, 'graph, T: Float> TensorBuilder<T> {
    #[inline]
    pub fn set_known_shape(mut self, s: Vec<isize>) -> TensorBuilder<T> {
        self.known_shape = Some(KnownShape::new(s));
        self
    }

    #[inline]
    pub fn set_shape(mut self, s: &Tensor<'tensor, 'graph, T>) -> TensorBuilder<T> {
        self.shape = Some(s.id());
        self
    }

    #[inline]
    pub fn set_differentiable(mut self, a: bool) -> TensorBuilder<T> {
        self.can_have_gradient = a;
        self
    }

    #[inline]
    pub fn set_inputs_raw(mut self, a: Vec<Input>) -> TensorBuilder<T> {
        self.inputs = a;
        self
    }

    #[inline]
    pub fn set_inputs(mut self, a: &[&Tensor<T>]) -> TensorBuilder<T> {
        self.inputs = a.into_iter().map(|&x| Input::new(x)).collect::<Vec<_>>();
        // TODO: append xs to to out_edges
        self
    }

    #[inline]
    pub fn set_input_raw(mut self, a: Vec<Input>) -> TensorBuilder<T> {
        self.inputs = a;
        self
    }

    #[inline]
    pub fn set_input_mut(mut self, val: &Tensor<T>) -> TensorBuilder<T> {
        self.inputs = vec![Input::new_mut(val)];
        self
    }

    #[inline]
    pub fn set_input(mut self, val: &Tensor<T>) -> TensorBuilder<T> {
        self.inputs = vec![Input::new(val)];
        self
    }

    #[inline]
    pub fn set_is_placeholder(mut self, a: bool) -> TensorBuilder<T> {
        self.is_placeholder = a;
        self
    }

    #[inline]
    pub fn set_constant_array(mut self, a: Arc<NdArray<T>>) -> TensorBuilder<T> {
        self.constant_array = Some(a);
        self
    }

    #[inline]
    pub fn set_variable_array(mut self, a: Arc<RwLock<NdArray<T>>>) -> TensorBuilder<T> {
        self.variable_array = Some(a);
        self
    }

    #[inline]
    pub fn set_input_indices(mut self, a: Vec<usize>) -> TensorBuilder<T> {
        self.input_indices = Some(a);
        self
    }

    #[inline]
    pub fn set_backprop_inputs(mut self, a: Vec<Input>) -> TensorBuilder<T> {
        self.backprop_inputs = Some(a);
        self
    }

    #[inline]
    pub fn build<O>(self, graph: &'graph Graph<T>, op: O) -> Tensor<'tensor, 'graph, T>
    where
        O: op::Op<T> + Sync + 'static,
    {
        let rank = if self.inputs.len() == 0 {
            0
        } else {
            self.inputs
                .iter()
                .map(|a| a.get(graph).tensor.top_rank)
                .max()
                .map(|a| a + 1)
                .unwrap_or(0)
        };

        let input_indices = if let Some(a) = self.input_indices {
            assert_eq!(
                a.len(),
                self.inputs.len(),
                "input_indices.len() must match inputs length"
            );
            a
        } else {
            vec![0; self.inputs.len()]
        };

        let new = TensorInternal {
            // `id` is set in `c.install`
            id: usize::default(),
            op: Box::new(op),
            in_edges: self.inputs,
            top_rank: rank,
            shape: self.shape,
            has_persistent_array: self.variable_array.is_some() || self.constant_array.is_some(),
            variable_array: self.variable_array,
            constant_array: self.constant_array,
            is_placeholder: self.is_placeholder,
            is_differentiable: self.can_have_gradient,
            input_indices,
            backprop_inputs: self.backprop_inputs,
            known_shape: self.known_shape,
        };
        Tensor {
            tensor: graph.install(new),
            graph,
        }
    }
}

pub(crate) struct Dummy;

impl<T: Float> crate::op::Op<T> for Dummy {
    fn name(&self) -> &str {
        "dummy"
    }

    fn compute(&self, _: &mut crate::op::OpComputeContext<T>) {
        unreachable!()
    }

    fn grad(&self, ctx: &mut OpGradientContext<T>) {
        let s = ctx.graph();
        let a = s.neg(ctx.output_grad());
        let ret = a + a;
        ctx.set_input_grads(vec![Some(ret)])
    }
}

// -- std::ops::{Add, Sub, Mul, Div} implementations --
macro_rules! impl_bin_op_between_tensor_and_float_trait {
    ($trt:ident, $func:ident, $op:ident) => {
        // Tensor op Float
        impl<'a, 'b: 'a, T: Float> $trt<T> for Tensor<'a, 'b, T> {
            type Output = Tensor<'a, 'b, T>;
            fn $func(self, rhs: T) -> Self::Output {
                self.graph
                    .$func(self.tensor, &self.graph.scalar(rhs).tensor)
            }
        }

        // &Tensor op Float
        impl<'l: 'a, 'a, 'b: 'a, T: Float> $trt<T> for &'l Tensor<'a, 'b, T> {
            type Output = Tensor<'a, 'b, T>;
            fn $func(self, rhs: T) -> Self::Output {
                self.graph
                    .$func(self.tensor, &self.graph.scalar(rhs).tensor)
            }
        }
    };
}

macro_rules! impl_bin_op_between_tensor_and_primitive {
    ($trt:ident, $func:ident, $op:ident, $scalar_type:ty) => {
        // primitive op Tensor
        impl<'r: 'a, 'a, 'b: 'a, T: Float> $trt<Tensor<'a, 'b, T>> for $scalar_type {
            type Output = Tensor<'a, 'b, T>;
            fn $func(self, rhs: Tensor<'a, 'b, T>) -> Self::Output {
                rhs.graph
                    .$func(rhs.graph.scalar(T::from(self).unwrap()).tensor, rhs.tensor)
            }
        }

        // primitive op &Tensor
        impl<'r: 'a, 'a, 'b: 'a, T: Float> $trt<&'r Tensor<'a, 'b, T>> for $scalar_type {
            type Output = Tensor<'a, 'b, T>;
            fn $func(self, rhs: &'r Tensor<'a, 'b, T>) -> Self::Output {
                rhs.graph
                    .$func(&rhs.graph.scalar(T::from(self).unwrap()).tensor, rhs.tensor)
            }
        }
    };
}

impl_bin_op_between_tensor_and_float_trait!(Add, add, AddOp);
impl_bin_op_between_tensor_and_float_trait!(Sub, sub, SubOp);
impl_bin_op_between_tensor_and_float_trait!(Mul, mul, MulOp);
impl_bin_op_between_tensor_and_float_trait!(Div, div, DivOp);

impl_bin_op_between_tensor_and_primitive!(Add, add, AddOp, f64);
impl_bin_op_between_tensor_and_primitive!(Sub, sub, SubOp, f64);
impl_bin_op_between_tensor_and_primitive!(Mul, mul, MulOp, f64);
impl_bin_op_between_tensor_and_primitive!(Div, div, DivOp, f64);

impl_bin_op_between_tensor_and_primitive!(Add, add, AddOp, f32);
impl_bin_op_between_tensor_and_primitive!(Sub, sub, SubOp, f32);
impl_bin_op_between_tensor_and_primitive!(Mul, mul, MulOp, f32);
impl_bin_op_between_tensor_and_primitive!(Div, div, DivOp, f32);

macro_rules! impl_bin_op_between_tensors {
    ($trt:ident, $func:ident, $op:ident) => {
        // Tensor op Tensor
        impl<'a, 'b: 'a, T: Float> $trt for Tensor<'a, 'b, T> {
            type Output = Tensor<'a, 'b, T>;
            fn $func(self, rhs: Tensor<'a, 'b, T>) -> Self::Output {
                self.graph.$func(self.tensor, rhs.tensor)
            }
        }

        // Tensor op &Tensor
        impl<'r: 'a, 'a, 'b: 'a, T: Float> $trt<&'r Tensor<'a, 'b, T>> for Tensor<'a, 'b, T> {
            type Output = Tensor<'a, 'b, T>;
            fn $func(self, rhs: &'r Tensor<'a, 'b, T>) -> Self::Output {
                self.graph.$func(self.tensor, rhs.tensor)
            }
        }

        // &Tensor op Tensor
        impl<'l: 'a, 'a, 'b: 'a, T: Float> $trt<Tensor<'a, 'b, T>> for &'l Tensor<'a, 'b, T> {
            type Output = Tensor<'a, 'b, T>;
            fn $func(self, rhs: Tensor<'a, 'b, T>) -> Self::Output {
                self.graph.$func(self.tensor, rhs.tensor)
            }
        }

        // &Tensor op &Tensor
        // lifetime of the two tensors are unrelated
        impl<'l: 'a, 'r: 'a, 'a, 'b: 'a, T: Float> $trt<&'r Tensor<'a, 'b, T>>
            for &'l Tensor<'a, 'b, T>
        {
            type Output = Tensor<'a, 'b, T>;
            fn $func(self, rhs: &'r Tensor<'a, 'b, T>) -> Self::Output {
                self.graph.$func(self.tensor, rhs.tensor)
            }
        }
    };
}

impl_bin_op_between_tensors!(Add, add, AddOp);
impl_bin_op_between_tensors!(Sub, sub, SubOp);
impl_bin_op_between_tensors!(Mul, mul, MulOp);
impl_bin_op_between_tensors!(Div, div, DivOp);

/// Implementors can be converted to `Tensor`.
pub trait AsTensor<'tensor, 'graph: 'tensor, T: Float> {
    fn as_tensor(&self, graph: &'graph Graph<T>) -> Tensor<'tensor, 'graph, T>;
}

impl<'graph: 'tensor, 'tensor, T: Float> AsTensor<'tensor, 'graph, T>
    for Tensor<'tensor, 'graph, T>
{
    fn as_tensor(&self, _: &'graph Graph<T>) -> Tensor<'tensor, 'graph, T> {
        *self
    }
}

macro_rules! impl_as_tensor_for_array {
    ($num_elems:expr) => {
        impl<'graph: 'tensor, 'tensor, T: Float, I: crate::Int> AsTensor<'tensor, 'graph, T>
            for [I; $num_elems]
        {
            fn as_tensor(&self, graph: &'graph Graph<T>) -> Tensor<'tensor, 'graph, T> {
                let vec = self
                    .iter()
                    .map(|&a| T::from(a).unwrap())
                    .collect::<Vec<T>>();

                // unwrap is safe
                let arr = NdArray::from_shape_vec(ndarray::IxDyn(&[self.len()]), vec).unwrap();
                graph.convert_to_tensor(arr)
            }
        }
    };
}

impl_as_tensor_for_array!(0);
impl_as_tensor_for_array!(1);
impl_as_tensor_for_array!(2);
impl_as_tensor_for_array!(3);
impl_as_tensor_for_array!(4);
impl_as_tensor_for_array!(5);
impl_as_tensor_for_array!(6);
impl_as_tensor_for_array!(7);
impl_as_tensor_for_array!(8);

/// Trait to create constant (persistent) tensors.
pub trait Constant<'tensor, 'scope: 'tensor, F: Float, Src: Sized> {
    /// Creates a (persistent) constant tensor from an `NdArray` or `Arc<NdArray>`.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use ndarray::{self, array, IxDyn, Ix1, Array};
    /// use autograd as ag;
    /// use ag::tensor::Constant;
    ///
    /// let v1: Array<f64, Ix1> = array![2.];
    /// let v2: Arc<Array<f64, IxDyn>> = Arc::new(array![2.].into_dyn());
    ///
    /// ag::with(|g| {
    ///    // instantiate from NdArray
    ///    let v1: ag::Tensor<f64> = g.constant(v1);
    ///    // instantiate from Arc<RwLock<NdArray>>
    ///    let v2: ag::Tensor<f64> = g.constant(v2);
    ///    let y: ag::Tensor<f64> = 3. * v1 * v2;
    ///
    ///    assert_eq!(12., y.eval(&[]).unwrap()[0]);
    /// });
    /// ```
    fn constant(&'scope self, arr: Src) -> Tensor<'tensor, 'scope, F>;
}

/// Trait to create variable (persistent) tensors.
pub trait Variable<'tensor, 'scope: 'tensor, F: Float, Src: Sized> {
    /// Creates a shared variable tensor from an `NdArray` or `Arc<RwLock<NdArray>>`.
    ///
    /// A shared variable can be mutated with gradient descent methods
    /// implemented in `autograd::gradient_descent_ops`.
    /// For the usages, see https://github.com/perrier1034/rust-autograd/tree/master/examples.
    ///
    /// ```
    /// use std::sync::{Arc, RwLock};
    /// use ndarray::{self, array, IxDyn, Ix1, Array};
    /// use autograd as ag;
    /// use ag::tensor::Variable;
    ///
    /// let v1: Array<f64, Ix1> = array![2.];
    /// let v2: Arc<RwLock<Array<f64, IxDyn>>> = ag::array::shared(array![2.]);
    ///
    /// ag::with(|g| {
    ///    // Instantiate from an NdArray
    ///    let v1: ag::Tensor<f64> = g.variable(v1);
    ///    // Instantiate from an Arc<RwLock<NdArray>>
    ///    let v2: ag::Tensor<f64> = g.variable(v2);
    ///    let y: ag::Tensor<f64> = 3. * v1 * v2;
    ///
    ///    assert_eq!(12., y.eval(&[]).unwrap()[0]);
    /// });
    /// ```
    fn variable(&'scope self, arr: Src) -> Tensor<'tensor, 'scope, F>;
}

// method overload 1
impl<'tensor, 'graph: 'tensor, F: Float>
    Constant<'tensor, 'graph, F, Arc<ndarray::Array<F, ndarray::IxDyn>>>
    for crate::graph::Graph<F>
{
    #[inline]
    fn constant(
        &'graph self,
        arr: Arc<ndarray::Array<F, ndarray::IxDyn>>,
    ) -> Tensor<'tensor, 'graph, F> {
        Tensor::builder()
            .set_constant_array(arr)
            .build(self, crate::ops::basic_source_ops::Const)
    }
}

// method overload 2
macro_rules! impl_constant_dim {
    ($d:ty) => {
        impl<'tensor, 'graph: 'tensor, F: Float> Constant<'tensor, 'graph, F, ndarray::Array<F, $d>>
            for crate::graph::Graph<F>
        {
            #[inline]
            fn constant(&'graph self, arr: ndarray::Array<F, $d>) -> Tensor<'tensor, 'graph, F> {
                Tensor::builder()
                    .set_constant_array(Arc::new(arr.into_dyn()))
                    .build(self, crate::ops::basic_source_ops::Const)
            }
        }
    };
}

// method overload 1
impl<'tensor, 'graph: 'tensor, F: Float>
    Variable<'tensor, 'graph, F, Arc<RwLock<ndarray::Array<F, ndarray::IxDyn>>>>
    for crate::graph::Graph<F>
{
    #[inline]
    fn variable(
        &'graph self,
        arr: Arc<RwLock<ndarray::Array<F, ndarray::IxDyn>>>,
    ) -> Tensor<'tensor, 'graph, F> {
        Tensor::builder()
            .set_variable_array(arr)
            .build(self, crate::ops::basic_source_ops::Variable)
    }
}

// method overload 2
macro_rules! impl_variable_dim {
    ($d:ty) => {
        impl<'tensor, 'graph: 'tensor, F: Float> Variable<'tensor, 'graph, F, ndarray::Array<F, $d>>
            for crate::graph::Graph<F>
        {
            #[inline]
            fn variable(&'graph self, arr: ndarray::Array<F, $d>) -> Tensor<'tensor, 'graph, F> {
                Tensor::builder()
                    .set_variable_array(Arc::new(RwLock::new(arr.into_dyn())))
                    .build(self, crate::ops::basic_source_ops::Variable)
            }
        }
    };
}

impl_constant_dim!(ndarray::Ix0);
impl_constant_dim!(ndarray::Ix1);
impl_constant_dim!(ndarray::Ix2);
impl_constant_dim!(ndarray::Ix3);
impl_constant_dim!(ndarray::Ix4);
impl_constant_dim!(ndarray::Ix5);
impl_constant_dim!(ndarray::Ix6);
impl_constant_dim!(ndarray::IxDyn);

impl_variable_dim!(ndarray::Ix0);
impl_variable_dim!(ndarray::Ix1);
impl_variable_dim!(ndarray::Ix2);
impl_variable_dim!(ndarray::Ix3);
impl_variable_dim!(ndarray::Ix4);
impl_variable_dim!(ndarray::Ix5);
impl_variable_dim!(ndarray::Ix6);
impl_variable_dim!(ndarray::IxDyn);
