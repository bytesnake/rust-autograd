use crate::tensor::Tensor;
use crate::Context;
use crate::Float;

macro_rules! impl_op {
    ($name:ident) => {
        pub struct $name;
        impl<'a, T: Float> crate::op::Op<'a, T> for $name {
            fn name(&self) -> &str {
                stringify!($name)
            }

            fn grad(
                &self,
                _: &'a Tensor<'a, T>,
                _: &[&'a Tensor<'a, T>],
                _: &'a Tensor<'a, T>,
                _: &mut Context<'a, T>
            ) -> Vec<Option<&'a Tensor<'a, T>>> {
                unreachable!()
            }

            fn compute(&self, _: &mut crate::runtime::OpComputeContext<T>) {
                unreachable!()
            }
        }
    };
}

impl_op!(Variable);
impl_op!(Const);
impl_op!(Placeholder);
