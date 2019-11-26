//! Module defining stochastic gradient descent optimizer.
use crate::tensor::{Input, Tensor};
use crate::Context;
use crate::Float;

struct SGDOp<T: Float> {
    pub lr: T,
}

impl<'a, T: Float> crate::op::Op<'a, T> for SGDOp<T> {
    fn name(&self) -> &str {
        "SGD"
    }

    fn compute(&self, ctx: &mut crate::runtime::OpComputeContext<T>) {
        ctx.input_mut(0).scaled_add(-self.lr, &ctx.input(1));
        ctx.push_output(Err(crate::op::ComputeException::NoOutput));
    }

    fn grad(&self, _: &'a Tensor<'a, T>, _: &[&'a Tensor<'a, T>], _: &'a Tensor<'a, T>, _: &mut crate::context::Context<'a, T>) -> Vec<Option<&'a Tensor<'a, T>>> {
        vec![None]
    }
}

/// Vanilla SGD optimizer
///
/// ```
/// extern crate autograd as ag;
///
/// let sgd = ag::gradient_descent_ops::SGD { lr: 0.1 };
/// // let update_ops = sgd.compute_updates(params, grads)
/// ```
///
/// See also https://github.com/raskr/rust-autograd/blob/master/examples/mlp_mnist.rs
pub struct SGD<T: Float> {
    /// Learning rate
    pub lr: T,
}

impl<'a, T: Float> SGD<T> {
    /// Creates ops to optimize `params` with SGD.
    ///
    /// Evaluated results of the return values will be `None`.
    pub fn compute_updates(
        &self,
        params: &[&'a Tensor<'a, T>],
        grads: &[&'a Tensor<'a, T>],
        c: &'a mut Context<'a, T>
    ) -> Vec<&'a Tensor<'a, T>> {
        params
            .into_iter()
            .zip(grads)
            .map(|(param, grad)| {
                Tensor::builder()
                    .set_inputs_mut(vec![
                        Input::new_mut(param),
                        Input::new(grad),
                    ])
                    .build(c, SGDOp { lr: self.lr })
            })
            .collect()
    }
}
