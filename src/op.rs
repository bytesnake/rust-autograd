//! Defining things related to `ag::op::Op`.
//!
use crate::arrayvec::ArrayVec;
use crate::tensor::{Tensor, ScopedTensor};
use crate::{Float, Scope};
use std::marker::PhantomData;
use std::any::type_name;

// Op can have multiple output arrays.
pub type ComputeResults<'v, T> = ArrayVec<[Result<crate::ArrRepr<'v, T>, ComputeException>; 16]>;

#[derive(Clone, Copy, Debug)]
/// This is an `exception`, not an error.
pub enum ComputeException {
    /// Computation finished correctly with no output
    NoOutput,
}

/// Operation trait. `Tensor` wraps trait-object of this.
///
/// # Implementing differentiable operations
///
/// Many of well-known ops are pre-defined in `ag::ops`, but you can also
/// implement custom ops by hand.
///
/// ```
/// extern crate ndarray;
/// extern crate arrayvec;
/// extern crate autograd as ag;
///
/// use autograd::scope::Scope;
/// type NdArray<T: ag::Float> = ndarray::Array<T, ndarray::IxDyn>;
///
/// // Implements `Op` trait for `Sigmoid`.
/// struct Sigmoid;
///
/// impl<T: ag::Float> ag::op::Op<T> for Sigmoid {
///
///     fn name(&self) -> &str
///     {
///         "Sigmoid"
///     }
///
///     // In this method, any errors caused by bad user-inputs should results in "panic".
///     // (`ag::op::ComputeException` represents an exception rather than an error.)
///     fn compute(
///         &self,
///         ctx: &mut ag::runtime::OpComputeContext<T>,
///     ) {
///         let x = &ctx.input(0);
///         // Use `ndarray::Array::mapv` for element-wise computation.
///         let half = T::from(0.5).unwrap();
///         let y = x.mapv(|a| ((a * half).tanh() * half) + half);
///         ctx.push_output(Ok(ag::ArrRepr::Owned(y)));
///     }
///
///     fn grad(&self, gy: &ag::Tensor<T>, xs: &[&ag::Tensor<T>], y: &ag::Tensor<T>, c: &Scope<T>)
///         -> Vec<Option<ag::Tensor<T>>>
///     {
///         // Symbolic gradient of `x`
///         let gx = gy * (y - ag::square(y));
///         vec![Some(gx)]
///     }
/// }
///
/// // Symbolic `sigmoid` function for end-user.
/// fn sigmoid<T: ag::Float>(x: &ag::Tensor<T>, c: &Scope<T>) -> &ag::Tensor<T>
/// {
///     ag::Tensor::builder()
///         .set_inputs(&[x])
///         .set_shape(x.shape())
///         .build(c, Sigmoid)
/// }
/// ```
pub trait Op<T: Float> {
    /// Name of this op
    fn name(&self) -> &str {
        type_name::<Self>()
    }

    /// Runs this op.
    fn compute(&self, ctx: &mut crate::runtime::OpComputeContext<T>);

    /// Returns symbolic gradients for input nodes by use of output gradient etc.
    ///
    /// # Arguments
    ///
    /// * `gy` - Symbolic representation of the gradient of `compute`'s return value
    /// * `xs` - Symbolic representation of `compute::xs`
    /// * `y` - Symbolic representation of `compute`'s return value
    ///
    /// NOTE:
    /// The number of return values must match `xs.len()`.
    fn grad(&self, ctx: &mut crate::gradient::GradientContext<T>);
}

pub struct DummyOp<F: Float> {
    pub phantom: PhantomData<F>
}

impl<F: Float> DummyOp<F> {
    pub fn new() -> Self {
        DummyOp {
            phantom: PhantomData
        }
    }
}

impl<T: Float> Op<T> for DummyOp<T> {

    fn name(&self) -> &str {
        "dummy"
    }

    /// Runs this op.
    fn compute(&self, ctx: &mut crate::runtime::OpComputeContext<T>) {}

    fn grad(&self, ctx: &mut crate::gradient::GradientContext<T>) {}
}