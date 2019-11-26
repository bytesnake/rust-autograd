//! Defining things related to `ag::op::Op`.
//!
use crate::tensor::Tensor;
use crate::Float;
use crate::arrayvec::ArrayVec;

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
/// use autograd::context::Context;
/// type NdArray<T: ag::Float> = ndarray::Array<T, ndarray::IxDyn>;
///
/// // Implements `Op` trait for `Sigmoid`.
/// struct Sigmoid;
///
/// impl<'a, T: ag::Float> ag::op::Op<T> for Sigmoid {
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
///     fn grad(&self, gy: &ag::Tensor<'a, T>, xs: &[&ag::Tensor<'a, T>], y: &ag::Tensor<'a, T>, c: &mut Context<'a, T>)
///         -> Vec<Option<ag::Tensor<'a, T>>>
///     {
///         // Symbolic gradient of `x`
///         let gx = gy * (y - ag::square(y));
///         vec![Some(gx)]
///     }
/// }
///
/// // Symbolic `sigmoid` function for end-user.
/// fn sigmoid<'a, T: ag::Float>(x: &ag::Tensor<'a, T>, c: &mut Context<'a, T>) -> &ag::Tensor<'a, T>
/// {
///     ag::Tensor::builder()
///         .set_inputs(&[x])
///         .set_shape(x.shape())
///         .build(c, Sigmoid)
/// }
/// ```
pub trait Op<'a, T: Float> {
    /// Name of this op
    fn name(&self) -> &str;

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
    fn grad(&self, gy: &'a Tensor<'a, T>, xs: &[&'a Tensor<'a, T>], y: &'a Tensor<'a, T>, c: &'a mut crate::context::Context<'a, T>) -> Vec<Option<&'a Tensor<'a, T>>>;
}
