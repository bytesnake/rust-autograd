#[allow(unused_imports)]
// Expose to prevent version conflict
#[macro_use(s)]
pub extern crate ndarray;
extern crate arrayvec;
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

pub mod gradient;
pub mod hook;
pub mod ndarray_ext;
pub mod op;
pub mod ops;
pub mod runtime;
pub mod scope;
pub mod tensor;
//pub mod test_helper;

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

//pub use crate::ops::gradient_descent_ops;

pub use crate::ndarray_ext::{NdArray, NdArrayView, NdArrayViewMut};

pub use crate::runtime::{eval, Eval, Feed};

pub use crate::tensor::Tensor;

pub use crate::ndarray_ext::ArrRepr;

pub use crate::scope::Scope;

#[inline]
#[doc(hidden)]
pub unsafe fn uninitialized_vec<T>(size: usize) -> Vec<T> {
    let mut buf = Vec::with_capacity(size);
    buf.set_len(size);
    buf
}

//#[inline]
//#[doc(hidden)]
//pub fn hook<T: Float>(hook: Hook<T>, node: &Tensor<T>) -> Tensor<T> {
//    let op = match hook {
//        Hook::Raw(func) => crate::ops::hook_ops::HookOp { func },
//        Hook::PrintShape => crate::ops::hook_ops::HookOp {
//            func: Box::new(|arr| println!("{:?}\n", arr.shape())),
//        },
//        Hook::Print => crate::ops::hook_ops::HookOp {
//            func: Box::new(|arr| println!("{:?}\n", arr)),
//        },
//    };
//    Tensor::builder().set_input(node).build(op)
//}

#[inline]
pub fn none_vec<T>(len: usize) -> Vec<Option<T>> {
    let mut vec = Vec::with_capacity(len);
    for _ in 0..len {
        vec.push(None);
    }
    vec
}
