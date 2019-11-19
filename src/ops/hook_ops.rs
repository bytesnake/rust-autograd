use crate::ndarray_ext::NdArrayView;
use crate::op;
use crate::tensor::Tensor;
use crate::Float;

pub struct Hook<T: Float> {
    pub name: Option<String>,
    pub func: Box<Fn(&NdArrayView<T>) -> () + Send + Sync>,
}

impl<T: Float> op::Op<T> for Hook<T> {
    fn name(&self) -> &str {
        "Hook"
    }

    fn compute(&self, ctx: &mut crate::runtime::OpComputeContext<T>) {
        if let Some(ref a) = self.name {
            println!("{}:", a);
        }
        let ret = ctx.input(0);
        (self.func)(&ret);
        ctx.set_output(vec![Ok(crate::ArrRepr::View(ret))]);
    }

    fn grad(&self, gy: &Tensor<T>, _: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![Some(gy.clone())]
    }
}
