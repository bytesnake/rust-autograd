//! Defining things related to `ag::Tensor`.
use crate::op;
//use crate::ops::binary_ops::{AddOp, DivOp, MulOp, SubOp};
use crate::Float;
use crate::Int;
use crate::NdArray;

use std::fmt;
use std::ops::Deref;
use std::ops::{Add, Div, Mul, Sub};
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

#[derive(Clone, Copy)]
pub struct ScopedTensor<'a, 'b: 'a, F: Float> {
    pub inner: &'a Tensor<F>,
    pub scope: &'b Scope<F>
}

impl<'a, 'b: 'a, T: Float> AsRef<ScopedTensor<'a, 'b, T>> for ScopedTensor<'a, 'b, T> {
    #[inline(always)]
    fn as_ref(&self) -> &ScopedTensor<'a, 'b, T> {
        self
    }
}

impl<'a, 'b: 'a, T: Float> Deref for ScopedTensor<'a, 'b, T> {
    type Target = Tensor<T>;
    fn deref(&self) -> &Self::Target {
        self.inner
    }
}

/// Symbolic multi-dimensional array.
pub struct Tensor<T: Float> {
    /// ID of this tensor
    pub(crate) id: usize,

    /// An operation to evaluate this tensor.
    pub op: Box<dyn op::Op<T> + Send + Sync>,

    /// References to immediate predecessors.
    pub inputs: Vec<Input<T>>,

    /// The rank number for topological ordering in a graph.
    pub top_rank: usize,

    /// *Symbolic* shape of this tensor.
    pub shape: Option<*const Tensor<T>>,

    /// An optional *persistent* NdArray.
    ///
    /// This is `Some` if this tensor is made from `ag::variable`.
    pub variable_array: Option<RwLock<NdArray<T>>>,

    /// An optional *persistent* NdArray.
    ///
    /// This is `Some` if this tensor is made from `ag::constant`.
    pub constant_array: Option<NdArray<T>>,

    /// This tensor is placeholder or not.
    pub is_placeholder: bool,

    /// This is `True` if this tensor can have gradient for any objectives.
    pub is_differentiable: bool,

    /// Input indices of arrays used in `compute`
    pub input_indices: Vec<usize>,

    /// Input nodes used when backprop.
    ///
    /// This is same as `inputs` in most cases.
    pub inputs_on_backprop: Option<Vec<Input<T>>>,

    /// Static shape of this tensor.
    /// Each dim size is *signed* for placeholders.
    pub known_shape: Option<KnownShape>,

    pub has_persistent_array: bool,
}

impl<T: Float> Tensor<T> {

    #[inline(always)]
    pub fn id(&self) -> usize {
        self.id
    }

    /// Returns a reference to the persistent array.
    ///
    /// Note that this is `Some` if this tensor derived from `ag::constant`; otherwise `None`
    #[inline]
    pub fn get_constant_array(&self) -> Option<&NdArray<T>> {
        if let Some(ref arr) = self.constant_array {
            Some(&*arr)
        } else {
            None
        }
    }

    /// Returns a mutable reference to the persistent array.
    ///
    /// Note that this is `Some` if this tensor derived from `ag::variable`; otherwise `None`.
    #[inline]
    pub fn get_variable_array(&self) -> Option<RwLockReadGuard<NdArray<T>>> {
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
    pub fn get_variable_array_mut(&self) -> Option<RwLockWriteGuard<NdArray<T>>> {
        if let Some(ref arr) = self.variable_array {
            Some(arr.write().unwrap())
        } else {
            None
        }
    }

    #[inline]
    pub fn clone_persistent_array(&self) -> Option<NdArray<T>> {
        if let Some(ref arr) = self.variable_array {
            Some((*arr.read().unwrap()).to_owned())
        } else {
            if let Some(ref arr) = self.constant_array {
                Some((*arr).clone())
            } else {
                None
            }
        }
    }

    #[inline]
    pub fn is_variable(&self) -> bool {
        self.variable_array.is_some()
    }

    #[inline]
    pub fn validate_feed_shape(&self, shape: &[usize]) {
        if !self.known_shape.as_ref().unwrap().validate(shape) {
            panic!(
                "Shape error: placeholder required {:?}, but got {:?}",
                self.known_shape.as_ref().unwrap().get(),
                shape
            );
        }
    }

    #[inline]
    pub fn builder() -> TensorBuilder<T> {
        TensorBuilder {
            shape: None,
            inputs: Vec::new(),
            can_have_gradient: true,
            constant_array: None,
            variable_array: None,
            is_placeholder: false,
            input_indices: None,
            inputs_on_backprop: None,
            known_shape: None,
        }
    }

    /// Evaluates this tensor as an ndarray object.
    ///
    /// ```
    /// use ndarray;
    /// use autograd as ag;
    ///
    /// let a = ag::zeros(&[2]);
    ///
    /// assert_eq!(a.eval(&[]), Some(ndarray::arr1(&[0., 0.]).into_dyn()));
    /// ```
    ///
    /// See also [eval](../fn.eval.html).
    pub fn eval<'k, 'v, 'b: 'k>(
        &'k self,
        feeds: &'v [crate::runtime::Feed<'k, 'v, T>],
        s: &'b Scope<T>
    ) -> Option<NdArray<T>> {
        crate::runtime::eval(&[self], feeds, s).remove(0)
    }

    #[doc(hidden)]
    #[inline]
    /// Returns true if this node has no incoming nodes.
    pub fn requires_compute(&self) -> bool {
        !self.is_placeholder && !self.has_persistent_array
    }

    #[doc(hidden)]
    #[inline]
    /// Returns true if this node has no incoming nodes.
    pub fn is_source(&self) -> bool {
        self.inputs.is_empty()
    }

    #[doc(hidden)]
    #[inline]
    pub fn get_backprop_inputs(&self) -> &[Input<T>] {
        self.inputs_on_backprop
            .as_ref()
            .unwrap_or(&self.inputs)
            .as_slice()
    }

    #[doc(hidden)]
    #[inline]
    pub fn get_scoped_input<'a, 'b: 'a>(&self, s: &'b Scope<T>) -> Vec<ScopedTensor<'a, 'b, T>> {
        let len = self.inputs.len();
        let mut ret = Vec::with_capacity(len);
        for a in self.inputs.iter() {
            ret.push(s.scope(a.get(s)));
        }
        ret
    }

    #[doc(hidden)]
    #[inline]
    pub fn get_backprop_inputs_ref(&self) -> &[Input<T>] {
        if let Some(ref a) = self.inputs_on_backprop {
            a.as_slice()
        } else {
            self.inputs.as_slice()
        }
    }

//    /// Registers a hook on a `Tensor`.
//    ///
//    /// Pre-defined hooks are descripted [here](../hook.html)
//    ///
//    /// ```
//    /// use autograd as ag;
//    ///
//    /// let a: ag::Tensor<f32> = ag::zeros(&[4, 2]).hook(ag::hook::Show);
//    /// let b: ag::Tensor<f32> = ag::ones(&[2, 3]).hook(ag::hook::ShowShape);
//    /// let c = ag::matmul(a, b);
//    ///
//    /// c.eval(&[]);
//    /// // [[0.0, 0.0],
//    /// // [0.0, 0.0],
//    /// // [0.0, 0.0],
//    /// // [0.0, 0.0]] shape=[4, 2], strides=[2, 1], layout=C (0x1)
//    ///
//    /// // [2, 3]
//    /// ```
//    #[inline]
//    pub fn hook<H: crate::hook::Hook<T> + Send + Sync + 'static>(
//        &mut self,
//        c: &mut Scope<T>,
//        hook: H,
//    ) -> &Tensor<T> {
//        Tensor::builder()
//            .set_input(self)
//            .build(c, crate::ops::hook_ops::HookOp::new(hook))
//    }
//
//    /// Shorthand for `Tensor::hook(ag::hook::Show)`
//    ///
//    /// See also [Tensor::hook](../tensor/struct.Tensor.html#method.hook)
//    ///
//    /// ```
//    /// use autograd as ag;
//    ///
//    /// let a: ag::Tensor<f32> = ag::zeros(&[4, 2]).show();
//    /// a.eval(&[]);
//    /// // [[0.0, 0.0],
//    /// // [0.0, 0.0],
//    /// // [0.0, 0.0],
//    /// // [0.0, 0.0]] shape=[4, 2], strides=[2, 1], layout=C (0x1)
//    /// ```
//    #[inline]
//    pub fn show(&mut self, c: &mut Scope<T>) -> &Tensor<T> {
//        self.hook(c, crate::hook::Show)
//    }
//
//    /// Shorthand for `Tensor::hook(ag::hook::Show)`
//    ///
//    /// See also [Tensor::hook](../tensor/struct.Tensor.html#method.hook)
//    ///
//    /// ```
//    /// use autograd as ag;
//    ///
//    /// let a: ag::Tensor<f32> = ag::zeros(&[4, 2]).show_with("My value:");
//    /// a.eval(&[]);
//    /// // My value:
//    /// // [[0.0, 0.0],
//    /// // [0.0, 0.0],
//    /// // [0.0, 0.0],
//    /// // [0.0, 0.0]] shape=[4, 2], strides=[2, 1], layout=C (0x1)
//    /// ```
//    #[inline]
//    pub fn show_with(&mut self, c: &mut Scope<T>, what: &'static str) -> &Tensor<T> {
//        self.hook(c, crate::hook::ShowWith(what))
//    }
//
//    /// Shorthand for `Tensor::hook(ag::hook::ShowShape)`
//    ///
//    /// See also [Tensor::hook](../tensor/struct.Tensor.html#method.hook)
//    ///
//    /// ```
//    /// use autograd as ag;
//    ///
//    /// let a: ag::Tensor<f32> = ag::zeros(&[2, 3]).show_shape();
//    /// a.eval(&[]);
//    /// // [2, 3]
//    /// ```
//    #[inline]
//    pub fn show_shape(&mut self, c: &mut Scope<T>) -> &Tensor<T> {
//        self.hook(c, crate::hook::ShowShape)
//    }
//
//    /// Shorthand for `Tensor::hook(ag::hook::ShowShape)`
//    ///
//    /// See also [Tensor::hook](../tensor/struct.Tensor.html#method.hook)
//    ///
//    /// ```
//    /// use autograd as ag;
//    ///
//    /// let a: ag::Tensor<f32> = ag::zeros(&[2, 3]).show_shape_with("My shape:");
//    /// a.eval(&[]);
//    /// // My shape:
//    /// // [2, 3]
//    /// ```
//    #[inline]
//    pub fn show_shape_with(&mut self, c: &mut Scope<T>, what: &'static str) -> &Tensor<T> {
//        self.hook(c, crate::hook::ShowShapeWith(what))
//    }
//
//    /// Shorthand for `Tensor::hook(ag::hook::PrintAny("what"))`
//    ///
//    /// See also [Tensor::hook](../tensor/struct.Tensor.html#method.hook)
//    ///
//    /// ```
//    /// use autograd as ag;
//    ///
//    /// let a: ag::Tensor<f32> = ag::zeros(&[2, 3]).show_shape();
//    /// a.eval(&[]);
//    /// // [2, 3]
//    /// ```
//    #[inline]
//    pub fn print(&mut self, c: &mut Scope<T>, what: &'static str) -> &Tensor<T> {
//        self.hook(c, crate::hook::PrintAny(what))
//    }
}

impl<T: Float> fmt::Debug for Tensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "[name: {}, num of inputs: {}]",
            self.op.name(),
            self.inputs.len()
        )
    }
}

// empty implementation
impl<T: Float> Eq for Tensor<T> {}

impl<T: Float> PartialEq for Tensor<T> {
    fn eq(&self, other: &Tensor<T>) -> bool {
        // compare addresses on the heap
        self.id() == other.id()
    }
}

use crate::scope::Scope;
use std::cell::UnsafeCell;
use std::hash::{Hash, Hasher};
use crate::gradient::GradientContext;

/// Raw pointer hashing
impl<T: Float> Hash for Tensor<T> {
    #[inline(always)]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state)
    }
}

impl<T: Float> AsRef<Tensor<T>> for Tensor<T> {
    #[inline(always)]
    fn as_ref(&self) -> &Tensor<T> {
        self
    }
}

impl<T: Float> fmt::Display for Tensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        unsafe {
            let input_names = self
                .inputs
                .iter()
                .map(|a| (&*a.val).op.name().to_string())
            .collect::<Vec<String>>();
            write!(
                f,
                "name={}, inputs={:?}",
                self.op.name(),
                input_names.as_slice()
            )
        }
    }
}

pub struct Input<T: Float> {
    pub val: *const Tensor<T>,
    pub mut_usage: bool,
}

impl<T: Float> Input<T> {
    #[inline(always)]
    pub fn new(val: &Tensor<T>) -> Input<T> {
        Input {
            val: val as *const _,
            mut_usage: false,
        }
    }

    #[inline]
    pub fn new_mut(val: &Tensor<T>) -> Input<T> {
        Input {
            val: val as *const _,
            mut_usage: true,
        }
    }

    pub fn get<'a, 'b: 'a>(&self, scope: &'b Scope<T>) -> &'a Tensor<T> {
        // UB doesn't occurs because the tensor is owned by a `Scope` object at this point.
        unsafe {
            scope.get((&*self.val).id())
        }
    }
}

/// Builder for `ag::Tensor`
pub struct TensorBuilder<T: Float> {
    shape: Option<*const Tensor<T>>,
    inputs: Vec<Input<T>>,
    can_have_gradient: bool,
    is_placeholder: bool,
    constant_array: Option<NdArray<T>>,
    variable_array: Option<RwLock<NdArray<T>>>,
    input_indices: Option<Vec<usize>>,
    inputs_on_backprop: Option<Vec<Input<T>>>,
    known_shape: Option<KnownShape>,
}

#[doc(hidden)]
pub struct KnownShape {
    shape: Vec<isize>,
    #[allow(dead_code)]
    is_fully_defined: bool,
}

impl KnownShape {
    #[inline]
    pub fn new(shape: Vec<isize>) -> Self {
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

    #[inline]
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
    pub fn is_fully_defined(&self) -> bool {
        self.is_fully_defined
    }
}

#[test]
fn test_build() {
    let ref a: Tensor<f32> = crate::zeros(&[4, 2]);
    let ref v: Tensor<f32> = crate::zeros(&[2, 3]);
    let ref b: Tensor<f32> = crate::zeros(&[4, 3]);
    let ref z = crate::matmul(a, v) + b;
    let mut vars = [a, v, b, z];
    // `sort_by_key` don't reverse the order of `a` and `v`
    vars.sort_by_key(|a| a.top_rank);
    assert_eq!(vars, [a, v, b, z])
}

impl<T: Float> TensorBuilder<T> {
    #[inline]
    pub fn set_known_shape(mut self, s: KnownShape) -> TensorBuilder<T> {
        self.known_shape = Some(s);
        self
    }

    #[inline]
    pub fn set_shape(mut self, s: &Tensor<T>) -> TensorBuilder<T> {
        self.shape = Some(s);
        self
    }

    #[inline]
    pub fn set_differentiable(mut self, a: bool) -> TensorBuilder<T> {
        self.can_have_gradient = a;
        self
    }

    #[inline]
    pub fn set_inputs_raw(mut self, a: Vec<Input<T>>) -> TensorBuilder<T> {
        self.inputs = a;
        self
    }

    #[inline]
    pub fn set_inputs_inner(mut self, a: &[&Tensor<T>]) -> TensorBuilder<T> {
        self.inputs = a.into_iter().map(|&b| Input::new(b)).collect::<Vec<_>>();
        self
    }

    #[inline]
    pub fn set_inputs(mut self, a: &[&ScopedTensor<T>]) -> TensorBuilder<T> {
        self.inputs = a.into_iter().map(|&b| Input::new(b.inner)).collect::<Vec<_>>();
        self
    }

    #[inline]
    pub fn set_input_raw(mut self, a: Vec<Input<T>>) -> TensorBuilder<T> {
        self.inputs = a;
        self
    }

    #[inline]
    pub fn set_input_mut(mut self, val: &ScopedTensor<T>, as_mut: bool) -> TensorBuilder<T> {
        self.inputs = vec![Input::new(val.inner)];
        self
    }

    #[inline]
    pub fn set_input(mut self, val: &ScopedTensor<T>) -> TensorBuilder<T> {
        self.inputs = vec![Input::new_mut(val.inner)];
        self
    }

    #[inline]
    pub fn set_is_placeholder(mut self, a: bool) -> TensorBuilder<T> {
        self.is_placeholder = a;
        self
    }

    #[inline]
    pub fn set_constant_array(mut self, a: NdArray<T>) -> TensorBuilder<T> {
        self.constant_array = Some(a);
        self
    }

    #[inline]
    pub fn set_variable_array(mut self, a: NdArray<T>) -> TensorBuilder<T> {
        self.variable_array = Some(RwLock::new(a));
        self
    }

    #[inline]
    pub fn set_input_indices(mut self, a: Vec<usize>) -> TensorBuilder<T> {
        self.input_indices = Some(a);
        self
    }

    #[inline]
    pub fn set_backprop_inputs(mut self, a: Vec<Input<T>>) -> TensorBuilder<T> {
        self.inputs_on_backprop = Some(a);
        self
    }

    #[inline]
    pub fn build<'a, 'b: 'a, O>(self, c: &'b Scope<T>, op: O) -> ScopedTensor<'a, 'b, T>
    where
        O: op::Op<T> + Send + Sync + 'static
    {
        let rank = if self.inputs.len() == 0 {
            0
        } else {
            self.inputs
                .iter()
                .map(|a| a.get(c).top_rank)
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

        let tensor = Tensor {
            // id is set in `c.install`
            id: 0,
            op: Box::new(op),
            inputs: self.inputs,
            top_rank: rank,
            shape: self.shape,
            has_persistent_array: self.variable_array.is_some() || self.constant_array.is_some(),
            variable_array: self.variable_array,
            constant_array: self.constant_array,
            is_placeholder: self.is_placeholder,
            is_differentiable: self.can_have_gradient,
            input_indices,
            inputs_on_backprop: self.inputs_on_backprop,
            known_shape: self.known_shape,
        };
        ScopedTensor {
            inner: c.install(tensor),
            scope: c
        }
    }
}

pub struct Dummy;

impl<T: Float> crate::op::Op<T> for Dummy {
    fn name(&self) -> &str {
        "dummy"
    }

    fn compute(&self, ctx: &mut crate::runtime::OpComputeContext<T>) {
        unreachable!()
    }

    fn grad(&self, ctx: &mut GradientContext<T>) {
        let s = ctx.scope();
        let a = s.neg(ctx.output_grad());
        let ret = a + a;
        ctx.set_input_grads(vec![Some(ret)])
    }
}


// -- std::ops::{Add, Sub, Mul, Div} implementations --
macro_rules! impl_bin_op_between_tensor_and_float_trait {
    ($trt:ident, $func:ident, $op:ident) => {
        // Tensor op Float
        impl<'a, 'b: 'a, T: Float> $trt<T> for ScopedTensor<'a, 'b, T> {
            type Output = ScopedTensor<'a, 'b, T>;
            fn $func(self, rhs: T) -> Self::Output {
                self.scope.$func(&self.scope.scalar(rhs).inner, self.inner)
            }
        }

        // &Tensor op Float
        impl<'l: 'a, 'a, 'b: 'a, T: Float> $trt<T> for &'l ScopedTensor<'a, 'b, T> {
            type Output = ScopedTensor<'a, 'b, T>;
            fn $func(self, rhs: T) -> Self::Output {
                self.scope.$func(&self.scope.scalar(rhs).inner, self.inner)
            }
        }
    };
}

macro_rules! impl_bin_op_between_tensor_and_primitive {
    ($trt:ident, $func:ident, $op:ident, $scalar_type:ty) => {
        // primitive op Tensor
        impl<'r: 'a, 'a, 'b: 'a, T: Float> $trt<ScopedTensor<'a, 'b, T>> for $scalar_type {
            type Output = ScopedTensor<'a, 'b, T>;
            fn $func(self, rhs: ScopedTensor<'a, 'b, T>) -> Self::Output {
                rhs.scope.$func(rhs.scope.scalar(T::from(self).unwrap()).inner, rhs.inner)
            }
        }

        // primitive op &Tensor
        impl<'r: 'a, 'a, 'b: 'a, T: Float> $trt<&'r ScopedTensor<'a, 'b, T>> for $scalar_type {
            type Output = ScopedTensor<'a, 'b, T>;
            fn $func(self, rhs: &'r ScopedTensor<'a, 'b, T>) -> Self::Output {
                rhs.scope.$func(&rhs.scope.scalar(T::from(self).unwrap()).inner, rhs.inner)
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
        impl<'a, 'b: 'a, T: Float> $trt for ScopedTensor<'a, 'b, T> {
            type Output = ScopedTensor<'a, 'b, T>;
            fn $func(self, rhs: ScopedTensor<'a, 'b, T>) -> Self::Output {
                self.scope.$func(self.inner, rhs.inner)
            }
        }

        // Tensor op &Tensor
        impl<'r: 'a, 'a, 'b: 'a, T: Float> $trt<&'r ScopedTensor<'a, 'b, T>> for ScopedTensor<'a, 'b, T> {
            type Output = ScopedTensor<'a, 'b, T>;
            fn $func(self, rhs: &'r ScopedTensor<'a, 'b, T>) -> Self::Output {
                self.scope.$func(self.inner, rhs.inner)
            }
        }

        // &Tensor op Tensor
        impl<'l: 'a, 'a, 'b: 'a, T: Float> $trt<ScopedTensor<'a, 'b, T>> for &'l ScopedTensor<'a, 'b, T> {
            type Output = ScopedTensor<'a, 'b, T>;
            fn $func(self, rhs: ScopedTensor<'a, 'b, T>) -> Self::Output {
                self.scope.$func(self.inner, rhs.inner)
            }
        }

        // &Tensor op &Tensor
        // lifetime of the two tensors are unrelated
        impl<'l: 'a, 'r: 'a, 'a, 'b: 'a, T: Float>
        $trt<&'r ScopedTensor<'a, 'b, T>> for &'l ScopedTensor<'a, 'b, T> {
            type Output = ScopedTensor<'a, 'b, T>;
            fn $func(self, rhs: &'r ScopedTensor<T>) -> Self::Output {
                self.scope.$func(self.inner, rhs.inner)
            }
        }
    };
}

impl_bin_op_between_tensors!(Add, add, AddOp);
impl_bin_op_between_tensors!(Sub, sub, SubOp);
impl_bin_op_between_tensors!(Mul, mul, MulOp);
impl_bin_op_between_tensors!(Div, div, DivOp);
