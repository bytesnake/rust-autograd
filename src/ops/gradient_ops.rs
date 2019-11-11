use crate::op;
use crate::tensor::Tensor;
use crate::Float;

pub struct StopGradient;

impl<T: Float> op::Op<T> for StopGradient {
    fn name(&self) -> &str {
        "StopGradient"
    }

    fn compute(
        &self,
        ctx: &mut crate::runtime::OpComputeContext<T>,
    ) {
        let ret = ctx.input(0);
        ctx.set_output(vec![Ok(crate::ArrRepr::View(ret))]);
    }

    fn grad(&self, _: &Tensor<T>, _: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![None]
    }
}
