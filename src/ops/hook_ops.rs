use crate::op;
use crate::Context;
use crate::tensor::Tensor;
use crate::Float;
use std::marker::PhantomData;

pub struct HookOp<T: Float, H: crate::hook::Hook<T>> {
    phantom: PhantomData<T>,
    pub hook: H,
}

impl<'a, T: Float, H: crate::hook::Hook<T>> HookOp<T, H> {
    #[inline]
    pub fn new(hook: H) -> Self {
        HookOp {
            phantom: PhantomData,
            hook,
        }
    }
}

impl<'a, T: Float, H: crate::hook::Hook<T>> op::Op<'a, T> for HookOp<T, H> {
    fn name(&self) -> &str {
        "Hook"
    }

    fn compute(&self, ctx: &mut crate::runtime::OpComputeContext<T>) {
        let ret = ctx.input(0);
        self.hook.call(&ret);
        ctx.push_output(Ok(crate::ArrRepr::View(ret)));
    }

    fn grad(&self, gy: &'a Tensor<'a, T>, _: &[&'a Tensor<'a, T>], _: &'a Tensor<'a, T>, c: &mut Context<'a, T>) -> Vec<Option<&'a Tensor<'a, T>>> {
        vec![Some(gy.clone())]
    }
}
