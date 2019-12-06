//! Module defining stochastic gradient descent optimizer.
use crate::tensor::{Input, Tensor, ScopedTensor};
use crate::Float;
use crate::Scope;

struct SGDOp<T: Float> {
    pub lr: T,
}

impl<T: Float> crate::op::Op<T> for SGDOp<T> {
    fn name(&self) -> &str {
        "SGD"
    }

    fn compute(&self, ctx: &mut crate::runtime::OpComputeContext<T>) {
        ctx.input_mut(0).scaled_add(-self.lr, &ctx.input(1));
        ctx.push_output(Err(crate::op::ComputeException::NoOutput));
    }

    fn grad(&self, ctx: &mut crate::gradient::GradientContext<T>) {
        ctx.set_input_grads(vec![None])
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

impl<'a, 'b: 'a, T: Float> SGD<T> {
    /// Creates ops to optimize `params` with SGD.
    ///
    /// Evaluated results of the return values will be `None`.
    pub fn compute_updates(
        &self,
        params: Vec<ScopedTensor<'a, 'b, T>>,
        grads: Vec<ScopedTensor<'a, 'b, T>>,
        c: &'b Scope<T>,
    ) -> Vec<ScopedTensor<'a, 'b, T>> {
        let len = params.len();
        let mut ret = Vec::with_capacity(len);
        for i in 0..len {
            ret.push(
                Tensor::builder()
                    .set_inputs_raw(vec![Input::new_mut(&params[i]), Input::new(&grads[i])])
                    .build(c, SGDOp { lr: self.lr }),
            );
        }
        ret
    }
}
