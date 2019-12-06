use crate::op;
use crate::tensor::{Tensor, ScopedTensor};
use crate::Float;
use crate::Scope;

pub struct StopGradient;

impl<T: Float> op::Op<T> for StopGradient {
    fn name(&self) -> &str {
        "StopGradient"
    }

    fn compute(&self, ctx: &mut crate::runtime::OpComputeContext<T>) {
        let ret = ctx.input(0);
        ctx.push_output(Ok(crate::ArrRepr::View(ret)));
    }

    fn grad(&self, ctx: &mut crate::gradient::GradientContext<T>) {
        ctx.set_input_grads(vec![None]);
    }
}
