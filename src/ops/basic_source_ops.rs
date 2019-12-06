use crate::tensor::{Tensor, ScopedTensor};
use crate::Float;
use crate::Scope;

macro_rules! impl_op {
    ($name:ident) => {
        pub struct $name;
        impl<T: Float> crate::op::Op<T> for $name {
            fn name(&self) -> &str {
                stringify!($name)
            }

            fn grad(&self, ctx: &mut crate::gradient::GradientContext<T>) {
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
