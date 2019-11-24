use super::*;

/// Trait for hook object on a tensor
/// ```
/// extern crate autograd as ag;
///
/// let a: ag::Tensor<f32> = ag::zeros(&[4, 2]).hook(ag::hook::Show);
/// let b: ag::Tensor<f32> = ag::ones(&[2, 3]).hook(ag::hook::ShowShape);
/// let c = ag::matmul(a, b);
///
/// c.eval(&[]);
/// // [[0.0, 0.0],
/// // [0.0, 0.0],
/// // [0.0, 0.0],
/// // [0.0, 0.0]] shape=[4, 2], strides=[2, 1], layout=C (0x1)
///
/// // [2, 3]
/// ```
pub trait Hook<T: Float> {
    fn call(&self, arr: &crate::ndarray::ArrayViewD<T>) -> ();
}

/// Print given string.
pub struct PrintAny(pub &'static str);

/// Show an array where this hook set.
pub struct Show;

/// Show an array where this hook set with any prefix.
pub struct ShowWith(pub &'static str);

/// Show an array's shape where this hook set.
pub struct ShowShape;

/// Show an array's shape where this hook set with any prefix.
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
