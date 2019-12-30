use crate::op;
use crate::Float;
use std::marker::PhantomData;

pub struct HookOp<T: Float, H: crate::hook::Hook<T>> {
    phantom: PhantomData<T>,
    pub hook: H,
}

impl<T: Float, H: crate::hook::Hook<T>> HookOp<T, H> {
    #[inline]
    pub fn new(hook: H) -> Self {
        HookOp {
            phantom: PhantomData,
            hook,
        }
    }
}

impl<T: Float, H: crate::hook::Hook<T>> op::Op<T> for HookOp<T, H> {
    fn compute(&self, ctx: &mut crate::op::OpComputeContext<T>) {
        let ret = ctx.input(0);
        self.hook.call(&ret);
        ctx.push_output(Ok(crate::ArrRepr::View(ret)));
    }

    fn grad(&self, ctx: &mut crate::op::OpGradientContext<T>) {
        ctx.set_input_grads(vec![Some(ctx.output_grad())])
    }
}
