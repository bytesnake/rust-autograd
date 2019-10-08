//! Defining things related to `ag::Tensor`.
use crate::binary_ops::{AddOp, DivOp, MulOp, SubOp};
use crate::op;
use crate::ops;
use crate::Float;
use crate::Int;
use crate::NdArray;

use std::cell::Cell;
use std::fmt;
use std::mem;
use std::ops::{Add, Div, Mul, Sub};
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, RwLock};

/// Symbolic multi-dimensional array.
pub struct Tensor<T: Float>(pub Arc<TensorCore<T>>);

#[doc(hidden)]
pub struct TensorCore<T: Float> {
    /// An operation to evaluate this tensor.
    pub op: Box<op::Op<T>>,

    /// References to immediate predecessors.
    pub inputs: Vec<Tensor<T>>,

    /// The rank number for topological ordering in a graph.
    pub top_rank: usize,

    /// *Symbolic* shape of this tensor.
    pub shape: Option<Tensor<T>>,

    /// An optional *persistent* NdArray.
    ///
    /// This is `Some` if this tensor is made from `ag::variable` or `ag::constant`.
    persistent_array: Option<PersistentArray<T>>,

    /// This tensor is placeholder or not.
    pub is_placeholder: bool,

    /// This is `True` if this tensor can have gradient for any objectives.
    pub is_differentiable: bool,

    /// Input indices of arrays used in `compute`
    pub input_indices: Vec<usize>,

    /// Input nodes used when backprop.
    ///
    /// This is same as `inputs` in most cases.
    pub inputs_on_backprop: Option<Vec<Tensor<T>>>,

    /// Static shape of this tensor.
    /// Each dim size is *signed* for placeholders.
    pub known_shape: Option<KnownShape>,
}

enum PersistentArray<T: Float> {
    Variable(NdArray<T>),
    Constant(NdArray<T>),
}

impl<T: Float> Tensor<T> {
    /// Returns a reference to the persistent array.
    ///
    /// Returns `Some` if this tensor is made from `ag::variable` or `ag::constant`.
    pub fn get_persistent_array(&self) -> Option<&NdArray<T>> {
        match self.persistent_array {
            Some(ref a) => match a {
                PersistentArray::Variable(ref arr) => Some(arr),
                PersistentArray::Constant(ref arr) => Some(arr),
            },
            None => None,
        }
    }

    /// Returns a mutable reference to the persistent array.
    ///
    /// Returns `Some` if this tensor is made from `ag::variable`.
    pub unsafe fn get_persistent_array_mut(&self) -> Option<&mut NdArray<T>> {
        mem::transmute(
            self.persistent_array
                .as_ref()
                .and_then(|inner| match inner {
                    PersistentArray::Variable(arr) => Some(arr),
                    PersistentArray::Constant(_) => None,
                }),
        )
    }

    /// Returns `True` if this tensor is made from `ag::variable` or `ag::constant`.
    #[inline]
    pub fn has_persistent_array(&self) -> bool {
        self.persistent_array.is_some()
    }
}

/// Builder for `ag::Tensor`
pub struct TensorBuilder<T: Float> {
    shape: Option<Tensor<T>>,
    inputs: Vec<Tensor<T>>,
    can_have_gradient: bool,
    is_placeholder: bool,
    persistent_array: Option<PersistentArray<T>>,
    input_indices: Option<Vec<usize>>,
    inputs_on_backprop: Option<Vec<Tensor<T>>>,
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
    pub fn set_shape(mut self, s: Tensor<T>) -> TensorBuilder<T> {
        self.shape = Some(s);
        self
    }

    #[inline]
    pub fn set_differentiable(mut self, a: bool) -> TensorBuilder<T> {
        self.can_have_gradient = a;
        self
    }

    #[inline]
    pub fn set_inputs(mut self, a: Vec<&Tensor<T>>) -> TensorBuilder<T> {
        self.inputs = a.iter().map(|b| (*b).clone()).collect::<Vec<Tensor<T>>>();
        self
    }

    #[inline]
    pub fn set_inputs_slice(mut self, a: &[&Tensor<T>]) -> TensorBuilder<T> {
        self.inputs = a.iter().map(|b| (*b).clone()).collect::<Vec<Tensor<T>>>();
        self
    }

    #[inline]
    pub fn set_input(mut self, a: &Tensor<T>) -> TensorBuilder<T> {
        self.inputs = vec![a.clone()];
        self
    }

    #[inline]
    pub fn set_is_placeholder(mut self, a: bool) -> TensorBuilder<T> {
        self.is_placeholder = a;
        self
    }

    #[inline]
    pub fn set_constant_array(mut self, a: NdArray<T>) -> TensorBuilder<T> {
        self.persistent_array = Some(PersistentArray::Constant(a));
        self
    }

    #[inline]
    pub fn set_variable_array(mut self, a: NdArray<T>) -> TensorBuilder<T> {
        self.persistent_array = Some(PersistentArray::Variable(a));
        self
    }

    #[inline]
    pub fn set_input_indices(mut self, a: Vec<usize>) -> TensorBuilder<T> {
        self.input_indices = Some(a);
        self
    }

    #[inline]
    pub fn set_backprop_inputs(mut self, a: Vec<Tensor<T>>) -> TensorBuilder<T> {
        self.inputs_on_backprop = Some(a);
        self
    }

    #[inline]
    pub fn build<O: op::Op<T> + 'static>(self, op: O) -> Tensor<T> {
        let rank = if self.inputs.len() == 0 {
            0
        } else {
            self.inputs
                .iter()
                .map(|a| a.top_rank)
                .max()
                .map(|a| a + 1)
                .unwrap_or(0)
        };

        let input_indices = if let Some(a) = self.input_indices {
            assert_eq!(a.len(), self.inputs.len());
            a
        } else {
            vec![0; self.inputs.len()]
        };

        Tensor(Arc::new(TensorCore {
            op: Box::new(op),
            inputs: self.inputs,
            top_rank: rank,
            shape: self.shape,
            persistent_array: self.persistent_array,
            is_placeholder: self.is_placeholder,
            is_differentiable: self.can_have_gradient,
            input_indices,
            inputs_on_backprop: self.inputs_on_backprop,
            known_shape: self.known_shape,
        }))
    }
}

impl<T: Float> Tensor<T> {
    #[inline]
    pub fn builder() -> TensorBuilder<T> {
        TensorBuilder {
            shape: None,
            inputs: Vec::new(),
            can_have_gradient: true,
            persistent_array: None,
            is_placeholder: false,
            input_indices: None,
            inputs_on_backprop: None,
            known_shape: None,
        }
    }

    /// Evaluates this tensor as an ndarray object.
    ///
    /// ```
    /// extern crate ndarray;
    /// extern crate autograd as ag;
    ///
    /// let a = ag::zeros(&[2]);
    ///
    /// assert_eq!(a.eval(&[]), Some(ndarray::arr1(&[0., 0.]).into_dyn()));
    /// ```
    ///
    /// See also [eval](../fn.eval.html).
    pub fn eval<'k, 'v>(&self, feeds: &'v [crate::runtime::Feed<'k, 'v, T>]) -> Option<NdArray<T>> {
        crate::runtime::eval(&[self], feeds).remove(0)
    }

    /// Returns the (symbolic) shape of this tensor.
    ///
    /// ```
    /// extern crate autograd as ag;
    ///
    /// let ref x: ag::Tensor<f32> = ag::zeros(&[2, 3]);
    /// let ref s = x.shape();
    ///
    /// assert_eq!(&[2., 3.], s.eval(&[]).unwrap().as_slice().unwrap());
    /// ```
    ///
    /// See also [shape](../ops/fn.shape.html).
    #[inline]
    pub fn shape(&self) -> Tensor<T> {
        crate::ops::shape(self)
    }

    /// Returns the (symbolic) rank of this tensor.
    ///
    /// ```
    /// extern crate ndarray;
    /// extern crate autograd as ag;
    ///
    /// let ref x: ag::Tensor<f32> = ag::zeros(&[2, 3, 4]);
    /// let ref r = x.rank();
    ///
    /// assert_eq!(3., r.eval(&[]).unwrap()[ndarray::IxDyn(&[])]);
    /// ```
    ///
    /// See also [rank](../ops/fn.rank.html).
    pub fn rank(&self) -> Tensor<T> {
        crate::ops::rank(self)
    }

    /// Returns the (symbolic) size of this tensor.
    ///
    ///
    /// ```
    /// extern crate ndarray;
    /// extern crate autograd as ag;
    ///
    /// let ref a: ag::Tensor<f32> = ag::zeros(&[4, 3]);
    /// let ref b = a.size();
    ///
    /// assert_eq!(12., b.eval(&[]).unwrap()[ndarray::IxDyn(&[])]);
    /// ```
    ///
    /// See also [size](../ops/fn.size.html).
    pub fn size(&self) -> Tensor<T> {
        crate::ops::size(self)
    }

    #[doc(hidden)]
    #[inline]
    /// Returns true if this node has no incoming nodes.
    pub fn is_source(&self) -> bool {
        self.inputs.is_empty()
    }

    #[doc(hidden)]
    #[inline]
    pub fn get_backprop_inputs(&self) -> &[Tensor<T>] {
        self.inputs_on_backprop
            .as_ref()
            .unwrap_or(&self.inputs)
            .as_slice()
    }

    #[doc(hidden)]
    #[inline]
    pub fn get_input_refs(&self) -> Vec<&Tensor<T>> {
        self.inputs.iter().map(|a| a).collect::<Vec<&Tensor<T>>>()
    }

    #[doc(hidden)]
    #[inline]
    pub fn get_backprop_inputs_ref(&self) -> Vec<&Tensor<T>> {
        self.inputs_on_backprop
            .as_ref()
            .unwrap_or(&self.inputs)
            .iter()
            .map(|a| a)
            .collect::<Vec<_>>()
    }

    /// Registers a hook on a `Tensor`.
    ///
    /// Pre-defined hooks are
    ///
    /// * Print - prints the evaluation result of this tensor. (See also [p](../tensor/struct.Tensor.html#method.p))
    /// * PrintShape - prints the evaluated shape of this tensor. (See also [ps](../tensor/struct.Tensor.html#method.ps))
    /// * Raw - executes a given closure (See also [with_fn](../tensor/struct.Tensor.html#method.with_fn))
    ///
    /// ```
    /// extern crate autograd as ag;
    ///
    /// let a: ag::Tensor<f32> = ag::zeros(&[4, 2]).with(ag::Hook::Print);
    /// let b: ag::Tensor<f32> = ag::ones(&[2, 3]).with(ag::Hook::PrintShape);
    /// let c = ag::matmul(a, b);
    ///
    /// c.eval(&[]);
    /// // Zeros:
    /// // [[0.0, 0.0],
    /// // [0.0, 0.0],
    /// // [0.0, 0.0],
    /// // [0.0, 0.0]] shape=[4, 2], strides=[2, 1], layout=C (0x1)
    ///
    /// // Shape of Ones:
    /// // [2, 3]
    /// ```
    #[inline]
    pub fn with(&self, hook: crate::Hook<T>) -> Tensor<T> {
        crate::hook(hook, self)
    }

    /// Registers a hook using closure on a `Tensor`.
    ///
    /// See also [with](../tensor/struct.Tensor.html#method.with)
    ///
    /// ```
    /// extern crate autograd as ag;
    ///
    /// let a: ag::Tensor<f32> = ag::ones(&[4, 2]);
    /// let b: ag::Tensor<f32> = ag::zeros(&[2, 3]);
    /// let c = ag::matmul(a, b).with_fn(Box::new(|arr| println!("My shape: {:?}", arr.shape())));
    ///
    /// c.eval(&[]);
    /// // My shape: [4, 3]
    /// ```
    #[inline]
    pub fn with_fn(&self, hook: Box<Fn(&crate::ndarray::ArrayViewD<T>) -> ()>) -> Tensor<T> {
        crate::hook(crate::Hook::Raw(hook), self)
    }

    /// Shorthand for `Tensor::with(ag::Hook::Print)`
    ///
    /// See [with](../tensor/struct.Tensor.html#method.with)
    ///
    /// ```
    /// extern crate autograd as ag;
    ///
    /// let a: ag::Tensor<f32> = ag::zeros(&[4, 2]).p();
    /// a.eval(&[]);
    /// // Zeros:
    /// // [[0.0, 0.0],
    /// // [0.0, 0.0],
    /// // [0.0, 0.0],
    /// // [0.0, 0.0]] shape=[4, 2], strides=[2, 1], layout=C (0x1)
    /// ```
    #[inline]
    pub fn p(&self) -> Tensor<T> {
        self.with(crate::Hook::Print)
    }

    /// Shorthand for `Tensor::with(ag::Hook::PrintShape)`
    ///
    /// See [with](../tensor/struct.Tensor.html#method.with)
    ///
    /// ```
    /// extern crate autograd as ag;
    ///
    /// let a: ag::Tensor<f32> = ag::zeros(&[2, 3]).ps();
    /// a.eval(&[]);
    /// // Shape of Zeros:
    /// // [2, 3]
    /// ```
    #[inline]
    pub fn ps(&self) -> Tensor<T> {
        self.with(crate::Hook::PrintShape)
    }
}

// empty implementation
impl<T: Float> Eq for Tensor<T> {}

impl<T: Float> PartialEq for Tensor<T> {
    fn eq(&self, other: &Tensor<T>) -> bool {
        // compare addresses on the heap
        Arc::ptr_eq(&self.0, &other.0)
    }
}

use std::hash::{Hash, Hasher};

/// Raw pointer hashing
impl<T: Float> Hash for Tensor<T> {
    #[inline(always)]
    fn hash<H: Hasher>(&self, state: &mut H) {
        (&*(self.0) as *const TensorCore<T>).hash(state);
    }
}

impl<T: Float> AsRef<Tensor<T>> for Tensor<T> {
    #[inline(always)]
    fn as_ref(&self) -> &Tensor<T> {
        self
    }
}

// data is not cloned; only reference count is incremented.
impl<T: Float> Clone for Tensor<T> {
    fn clone(&self) -> Tensor<T> {
        Tensor(self.0.clone())
    }
}

impl<T: Float> Deref for Tensor<T> {
    type Target = Arc<TensorCore<T>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: Float> DerefMut for Tensor<T> {
    fn deref_mut(&mut self) -> &mut Arc<TensorCore<T>> {
        &mut self.0
    }
}

impl<T: Float> fmt::Display for Tensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let input_names = self
            .0
            .inputs
            .iter()
            .map(|a| a.op.name().to_string())
            .collect::<Vec<String>>();
        write!(
            f,
            "name={}, inputs={:?}",
            self.0.op.name(),
            input_names.as_slice()
        )
    }
}

/// Implementors can be converted to `Tensor`.
pub trait ArrayLike<T: Float> {
    fn as_tensor(&self) -> Tensor<T>;
}

impl<T: Float> ArrayLike<T> for Tensor<T> {
    fn as_tensor(&self) -> Tensor<T> {
        self.clone()
    }
}

macro_rules! impl_array_like_for_array {
    ($num_elems:expr) => {
        impl<T: Float, I: Int> ArrayLike<T> for [I; $num_elems] {
            fn as_tensor(&self) -> Tensor<T> {
                let vec = self
                    .iter()
                    .map(|&a| T::from(a).unwrap())
                    .collect::<Vec<T>>();

                // unwrap is safe
                let arr = NdArray::from_shape_vec(ndarray::IxDyn(&[self.len()]), vec).unwrap();
                ops::convert_to_tensor(arr)
            }
        }
    };
}

impl_array_like_for_array!(0);
impl_array_like_for_array!(1);
impl_array_like_for_array!(2);
impl_array_like_for_array!(3);
impl_array_like_for_array!(4);
impl_array_like_for_array!(5);
impl_array_like_for_array!(6);
impl_array_like_for_array!(7);
impl_array_like_for_array!(8);

// -- std::ops::{Add, Sub, Mul, Div} implementations --
macro_rules! impl_bin_op_between_tensor_and_float_trait {
    ($trt:ident, $func:ident, $op:ident) => {
        // Tensor op Float
        impl<T: Float> $trt<T> for Tensor<T> {
            type Output = Tensor<T>;
            fn $func(self, rhs: T) -> Self::Output {
                Tensor::builder()
                    .set_inputs(vec![&self, &ops::scalar(rhs)])
                    .set_shape(self.shape())
                    .build(crate::binary_ops::$op)
            }
        }

        // &Tensor op Float
        impl<'a, T: Float> $trt<T> for &'a Tensor<T> {
            type Output = Tensor<T>;
            fn $func(self, rhs: T) -> Self::Output {
                Tensor::builder()
                    .set_inputs(vec![&self, &ops::scalar(rhs)])
                    .set_shape(self.shape())
                    .build(crate::binary_ops::$op)
            }
        }
    };
}

macro_rules! impl_bin_op_between_tensor_and_primitive {
    ($trt:ident, $func:ident, $op:ident, $scalar_type:ty) => {
        // primitive op Tensor
        impl<T: Float> $trt<Tensor<T>> for $scalar_type {
            type Output = Tensor<T>;
            fn $func(self, rhs: Tensor<T>) -> Self::Output {
                Tensor::builder()
                    .set_inputs(vec![&ops::scalar(T::from(self).unwrap()), &rhs])
                    .set_shape(rhs.shape())
                    .build($op)
            }
        }

        // primitive op &Tensor
        impl<'a, T: Float> $trt<&'a Tensor<T>> for $scalar_type {
            type Output = Tensor<T>;
            fn $func(self, rhs: &'a Tensor<T>) -> Self::Output {
                Tensor::builder()
                    .set_inputs(vec![&ops::scalar(T::from(self).unwrap()), &rhs])
                    .set_shape(rhs.shape())
                    .build($op)
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
        impl<T: Float> $trt for Tensor<T> {
            type Output = Tensor<T>;
            fn $func(self, rhs: Tensor<T>) -> Self::Output {
                ops::$func(&self, &rhs)
            }
        }

        // Tensor op &Tensor
        impl<'a, T: Float> $trt<&'a Tensor<T>> for Tensor<T> {
            type Output = Tensor<T>;
            fn $func(self, rhs: &Tensor<T>) -> Self::Output {
                ops::$func(&self, rhs)
            }
        }

        // &Tensor op Tensor
        impl<'a, T: Float> $trt<Tensor<T>> for &'a Tensor<T> {
            type Output = Tensor<T>;
            fn $func(self, rhs: Tensor<T>) -> Self::Output {
                ops::$func(&self, &rhs)
            }
        }

        // &Tensor op &Tensor
        // lifetime of the two tensors are unrelated
        impl<'a, 'b, T: Float> $trt<&'a Tensor<T>> for &'b Tensor<T> {
            type Output = Tensor<T>;
            fn $func(self, rhs: &Tensor<T>) -> Self::Output {
                ops::$func(self, rhs)
            }
        }
    };
}

impl_bin_op_between_tensors!(Add, add, AddOp);
impl_bin_op_between_tensors!(Sub, sub, SubOp);
impl_bin_op_between_tensors!(Mul, mul, MulOp);
impl_bin_op_between_tensors!(Div, div, DivOp);
