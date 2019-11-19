#[allow(unused_imports)]
// Expose to prevent version conflict
#[macro_use(s)]
pub extern crate ndarray;
extern crate crossbeam;
extern crate hashbrown;
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;
extern crate libc;
#[cfg(not(feature = "mkl"))]
extern crate matrixmultiply;
extern crate num;
extern crate num_traits;
extern crate rand;
extern crate rayon;
extern crate rustc_hash;

#[macro_use]
#[doc(hidden)]
pub mod test_helper;

pub mod tensor;

#[doc(hidden)]
pub mod runtime;

#[doc(hidden)]
pub mod gradient;

pub mod ops;

pub mod ndarray_ext;

pub mod op;

use rustc_hash::FxHasher;
use std::any::TypeId;
use std::fmt;
use std::hash::BuildHasherDefault;

pub type FxHashMap<K, V> = hashbrown::HashMap<K, V, BuildHasherDefault<FxHasher>>;
pub type FxHashSet<V> = hashbrown::HashSet<V, BuildHasherDefault<FxHasher>>;

/// Primitive type in this crate, which is actually a decorated `num_traits::Float`.
pub trait Float:
    num_traits::Float
    + num_traits::NumAssignOps
    + Copy
    + Send
    + Sync
    + fmt::Display
    + fmt::Debug
    + Sized
    + 'static
{
}

#[doc(hidden)]
/// Internal trait.
pub trait Int:
    num::Integer
    + num_traits::NumAssignOps
    + num_traits::ToPrimitive
    + Copy
    + Send
    + fmt::Display
    + Sized
    + 'static
{
}

impl<T> Float for T where
    T: num::Float
        + num_traits::NumAssignOps
        + Copy
        + Send
        + Sync
        + fmt::Display
        + fmt::Debug
        + Sized
        + 'static
{
}

impl<T> Int for T where
    T: num::Integer
        + num_traits::NumAssignOps
        + num_traits::ToPrimitive
        + Copy
        + Send
        + Sync
        + fmt::Display
        + Sized
        + 'static
{
}

#[doc(hidden)]
#[inline(always)]
/// Return `true` if `A` and `B` are the same type
pub fn same_type<A: 'static, B: 'static>() -> bool {
    TypeId::of::<A>() == TypeId::of::<B>()
}

pub use crate::ndarray_ext::array_gen;

pub use crate::ops::*;

pub use crate::ops::gradient_descent_ops;

pub use crate::ndarray_ext::{NdArray, NdArrayView, NdArrayViewMut};

pub use crate::runtime::{eval, Eval, Feed};

pub use crate::tensor::Tensor;

pub use crate::ndarray_ext::ArrRepr;

#[inline]
#[doc(hidden)]
pub unsafe fn uninitialized_vec<T>(size: usize) -> Vec<T> {
    let mut buf = Vec::with_capacity(size);
    buf.set_len(size);
    buf
}

/// Registers a hook on a `Tensor`.
///
/// Pre-defined hooks are
///
/// * Print - prints the evaluation result of this tensor. (See also [p](tensor/struct.Tensor.html#method.p))
/// * PrintShape - prints the evaluated shape of this tensor. (See also [ps](tensor/struct.Tensor.html#method.ps))
/// * Raw - executes a given closure (See also [with_fn](tensor/struct.Tensor.html#method.with_fn))
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
pub enum Hook<T: Float> {
    Raw(Box<dyn Fn(&crate::ndarray_ext::NdArrayView<T>) -> () + Send + Sync>),
    Print,
    PrintShape,
}

// Use `Tensor::with`.
#[inline]
#[doc(hidden)]
pub fn hook<T: Float>(hook: Hook<T>, node: &Tensor<T>) -> Tensor<T> {
    let op = match hook {
        Hook::Raw(func) => crate::ops::hook_ops::Hook { func, name: None },
        Hook::PrintShape => crate::ops::hook_ops::Hook {
            func: Box::new(|arr| println!("{:?}\n", arr.shape())),
            name: Some(format!("Shape of {}", node.op.name())),
        },
        Hook::Print => crate::ops::hook_ops::Hook {
            func: Box::new(|arr| println!("{:?}\n", arr)),
            name: Some(node.op.name().to_owned()),
        },
    };
    Tensor::builder().set_input(node).build(op)
}

#[inline]
pub fn none_vec<T>(len: usize) -> Vec<Option<T>> {
    let mut vec = Vec::with_capacity(len);
    for _ in 0..len {
        vec.push(None);
    }
    vec
}
