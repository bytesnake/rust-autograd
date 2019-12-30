use crate::op;
use crate::Float;

pub struct StopGradient;

impl<T: Float> op::Op<T> for StopGradient {
    fn compute(&self, ctx: &mut crate::op::OpComputeContext<T>) {
        let ret = ctx.input(0);
        ctx.push_output(Ok(crate::ArrRepr::View(ret)));
    }

    fn grad(&self, ctx: &mut crate::op::OpGradientContext<T>) {
        ctx.set_input_grads(vec![None]);
    }
}
