//! Defining things used with `Tensor::hook`.
use super::*;

/// Trait for hook object on a tensor
///
/// ```
/// extern crate autograd as ag;
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
pub trait Hook<T: Float> {
    /// Calls this hook with the value of the tensor where this hook is set.
    fn call(&self, arr: &crate::ndarray::ArrayViewD<T>) -> ();
}

/// Prints a given string.
///
/// See also  [Tensor::print_any](../tensor/struct.Tensor.html#method.print_any)
///
/// ```
/// use autograd as ag;
/// use autograd::hook::PrintAny;
///
/// ag::with(|g| {
///     let a: ag::Tensor<f32> = g.zeros(&[2, 3]).hook(PrintAny("This is `a`"));
///     a.eval(&[]);
///     // This is `a`
/// });
/// ```
pub struct PrintAny(pub &'static str);

/// Shows an array where this hook is set.
///
/// See also  [Tensor::show](../tensor/struct.Tensor.html#method.show)
///
/// ```
/// use autograd as ag;
/// use ag::hook::Show;
///
/// ag::with(|g| {
///     let a: ag::Tensor<f32> = g.zeros(&[4, 2]).hook(Show);
///     a.eval(&[]);
///     // [[0.0, 0.0],
///     // [0.0, 0.0],
///     // [0.0, 0.0],
///     // [0.0, 0.0]] shape=[4, 2], strides=[2, 1], layout=C (0x1)
///     });
/// ```
pub struct Show;

/// Shows an array where this hook is set with any prefix.
///
/// See also  [Tensor::show_with](../tensor/struct.Tensor.html#method.show_with)
///
/// ```
/// use autograd as ag;
/// use ag::hook::ShowWith;
///
/// ag::with(|g| {
///     let a: ag::Tensor<f32> = g.zeros(&[4, 2]).hook(ShowWith("My value:"));
///     a.eval(&[]);
///     // My value:
///     // [[0.0, 0.0],
///     // [0.0, 0.0],
///     // [0.0, 0.0],
///     // [0.0, 0.0]] shape=[4, 2], strides=[2, 1], layout=C (0x1)
/// });
///
/// ```
pub struct ShowWith(pub &'static str);

/// Shows an array's shape where this hook is set.
///
/// See also  [Tensor::show_shape](../tensor/struct.Tensor.html#method.show_shape)
///
/// ```
/// use autograd as ag;
/// use ag::hook::ShowShape;
///
/// ag::with(|g| {
///     let a: ag::Tensor<f32> = g.zeros(&[2, 3]).hook(ShowShape);
///     a.eval(&[]);
///     // [2, 3]
/// });
/// ```
pub struct ShowShape;

/// Shows an array's shape where this hook is set with any prefix.
///
/// See also  [Tensor::show_shape_with](../tensor/struct.Tensor.html#method.show_shape_with)
///
/// ```
/// use autograd as ag;
/// use autograd::hook::ShowShapeWith;
///
/// ag::with(|g| {
///     let a: ag::Tensor<f32> = g.zeros(&[2, 3]).hook(ShowShapeWith("My shape:"));
///     a.eval(&[]);
///     // My shape:
///     // [2, 3]
/// });
/// ```
pub struct ShowShapeWith(pub &'static str);

impl<T: Float> Hook<T> for PrintAny {
    fn call(&self, _: &crate::ndarray_ext::NdArrayView<T>) -> () {
        println!("{}\n", self.0);
    }
}

impl<T: Float> Hook<T> for Show {
    fn call(&self, arr: &crate::ndarray_ext::NdArrayView<T>) -> () {
        println!("{:?}\n", arr);
    }
}

impl<T: Float> Hook<T> for ShowWith {
    fn call(&self, arr: &crate::ndarray_ext::NdArrayView<T>) -> () {
        println!("{}\n {:?}\n", self.0, arr);
    }
}

impl<T: Float> Hook<T> for ShowShape {
    fn call(&self, arr: &crate::ndarray_ext::NdArrayView<T>) -> () {
        println!("{:?}\n", arr.shape());
    }
}

impl<T: Float> Hook<T> for ShowShapeWith {
    fn call(&self, arr: &crate::ndarray_ext::NdArrayView<T>) -> () {
        println!("{}\n{:?}\n", self.0, arr.shape());
    }
}
