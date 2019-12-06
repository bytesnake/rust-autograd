use crate::{Float, Tensor};
use std::cell::UnsafeCell;

pub struct Scope<F: Float> {
    node_set: UnsafeCell<Vec<Tensor<F>>>,
}

impl<F: Float> Scope<F> {
    pub(crate) fn install(&self, mut node: Tensor<F>) -> &Tensor<F> {
        unsafe {
            let inner = &mut *self.node_set.get();
            let id = inner.len();
            node.id = id;
            inner.push(node);
            &inner[id]
        }
    }

    pub(crate) fn get(&self, i: usize) -> &Tensor<F> {
        unsafe {
            let inner = &*self.node_set.get();
            &inner[i]
        }
    }
}

/// The only way to make scopes
pub fn scope<'a, FN, F>(mut f: FN)
where
    F: Float,
    FN: FnMut(&Scope<F>) -> (),
{
    let s = Scope {
        node_set: UnsafeCell::new(Vec::with_capacity(128)),
    };
    f(&s);
}

// scope とは別の何かがさらにひつよう？
//pub fn example() {
//    // 普通に考えたら、Scope の中の ref たちは、scope より長生きしないといけない。
//
//    // 普通、内部に ref を持つ構造体を定義するとする。
//    // ここで s を借用する。
//    {
//        let arr = ndarray::arr1(&[1f32, 2f32]);
//        let s: Scope<f32> = Scope {
//            node_set: UnsafeCell::new(Vec::new()),
//        };
//        s.variable(arr);
//    }
//    // tensor に lifetime が付いてるのが悪いと思う。
//    //
//    // 1. 内部に inputs を tensor として持つには、ポインタしかない。
//    // *const として持つ。&*ptr すれば、&Tensor には変換できる。
//    // Tensor は内部に Tensor への参照でなく、id を持つようにするのは？
//    // id があれば、context で O(1) でアクセス可能。
//
//    //    scope(|s: Scope<'a, f32>| {
//    //        let a: &'a Tensor<'a, f32> = s.variable(ndarray::arr1(&[1f32, 2f32]));  // この s.variable の戻り値が、'a じゃなくて '_ になってるぽい
//    //        let b = s.variable(ndarray::arr1(&[1f32, 2f32]));
//    ////        let b = s.standard_normal(s.newshape(&[3, 2]));
//    //        let c = s.matmul(a, b);
//    //    });
//
//    // しかし、s はここで死ぬ。しかし、借用はまだ続いている？
//    // 正確に言うと、同時に終わる。
//}
