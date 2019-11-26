use crate::op;
use crate::Context;
use crate::tensor::Tensor;
use crate::Float;

pub struct StopGradient;

impl<'a, T: Float> op::Op<'a, T> for StopGradient {
    fn name(&self) -> &str {
        "StopGradient"
    }

    fn compute(&self, ctx: &mut crate::runtime::OpComputeContext<T>) {
        let ret = ctx.input(0);
        ctx.push_output(Ok(crate::ArrRepr::View(ret)));
    }

    fn grad(&self, _: &'a Tensor<'a, T>, _: &[&'a Tensor<'a, T>], _: &'a Tensor<'a, T>, c: &mut Context<'a, T>) -> Vec<Option<&'a Tensor<'a, T>>> {
        vec![None]
    }
}
