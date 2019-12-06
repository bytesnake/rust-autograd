use crate::ndarray_ext::{NdArray, NdArrayView};
use crate::op;
use crate::tensor::{Tensor, ScopedTensor};
use crate::Float;
use crate::Scope;
use ndarray;

pub struct ELU<T: Float> {
    pub alpha: T,
}

pub struct ELUGrad<T: Float> {
    pub alpha: T,
}

pub struct Identity;

pub struct ReLU;

pub struct Sigmoid;

pub struct Softplus;

pub struct Softmax {
    pub axis: isize,
}

#[inline]
pub fn softmax_forward<T: Float>(x: &NdArrayView<T>, axis: isize) -> NdArray<T> {
    let axis = if axis < 0 {
        (x.ndim() as isize + axis) as usize
    } else {
        axis as usize
    };

    let mut a = x.shape().to_vec();
    a[axis] = 1;
    let reduced_shape = a.as_slice();
    let max_fn = T::max;
    // unwrap is safe
    let ref max = x
        .fold_axis(ndarray::Axis(axis), T::min_value(), move |&a, &b| {
            max_fn(a, b)
        })
        .into_shape(ndarray::IxDyn(reduced_shape))
        .unwrap();
    // subtract `max` to prevent overflow
    let mut tmp = x - max;
    tmp.mapv_inplace(|a| a.exp());
    // unwrap is safe
    let sum = tmp
        .sum_axis(ndarray::Axis(axis))
        .into_shape(ndarray::IxDyn(reduced_shape))
        .unwrap();
    tmp /= &sum;
    tmp
}

impl<T: Float> op::Op<T> for Softmax {
    fn name(&self) -> &str {
        "Softmax"
    }

    fn compute(&self, ctx: &mut crate::runtime::OpComputeContext<T>) {
        let ret = Ok(crate::ArrRepr::Owned(softmax_forward(
            &ctx.input(0),
            self.axis,
        )));
        ctx.push_output(ret)
    }

    fn grad(&self, ctx: &mut crate::gradient::GradientContext<T>) {
        let s = ctx.scope();
        let y = &ctx.output();
        let gy = &ctx.output_grad();
        let sum = s.reduce_sum(&(y * gy), s.axes(&[self.axis]), true);
        ctx.set_input_grads(vec![Some((gy - sum) * y)]);
    }
}

impl<T: Float> op::Op<T> for Softplus {
    fn name(&self) -> &str {
        "Softplus"
    }

    fn compute(&self, ctx: &mut crate::runtime::OpComputeContext<T>) {
        use std::f64;
        let e = T::from(f64::consts::E).unwrap();
        let ret = Ok(crate::ArrRepr::Owned(
            ctx.input(0).map(move |a| (a.exp() + T::one()).log(e)),
        ));
        ctx.push_output(ret)
    }

    fn grad(&self, ctx: &mut crate::gradient::GradientContext<T>) {
        let gy = &ctx.output_grad();
        let s = ctx.scope();
        let a = s.exp(&ctx.input(0));
        let b = a + s.scalar(T::one());
        let gx = gy * (a / b);
        ctx.set_input_grads(vec![Some(gx)]);
    }
}

impl<T: Float> op::Op<T> for Sigmoid {
    fn name(&self) -> &str {
        "Sigmoid"
    }

    fn compute(&self, ctx: &mut crate::runtime::OpComputeContext<T>) {
        let half = T::from(0.5).unwrap();
        let ret = Ok(crate::ArrRepr::Owned(
            ctx.input(0)
                .mapv(move |a| ((a * half).tanh() * half) + half),
        ));
        ctx.push_output(ret)
    }

    fn grad(&self, ctx: &mut crate::gradient::GradientContext<T>) {
        let gy = ctx.output_grad();
        let y = ctx.output();
        let s = ctx.scope();
        ctx.set_input_grads(vec![Some(gy * (y - s.square(y)))]);
    }
}

impl<T: Float> op::Op<T> for ReLU {
    fn name(&self) -> &str {
        "ReLU"
    }

    fn compute(&self, ctx: &mut crate::runtime::OpComputeContext<T>) {
        let ret = Ok(crate::ArrRepr::Owned(
            ctx.input(0).map(|a| a.max(T::zero())),
        ));
        ctx.push_output(ret);
    }

    fn grad(&self, ctx: &mut crate::gradient::GradientContext<T>) {
        let s = ctx.scope();
        let gy = &ctx.output_grad();
        let bin = s.greater(&ctx.input(0), &s.scalar(T::zero()));
        ctx.set_input_grads(vec![Some(s.mul(&bin, &gy))]);
    }
}

impl<T: Float> op::Op<T> for Identity {
    fn name(&self) -> &str {
        "Identity"
    }

    fn compute(&self, ctx: &mut crate::runtime::OpComputeContext<T>) {
        // do nothing
        let ret = Ok(crate::ArrRepr::View(ctx.input(0)));
        ctx.push_output(ret)
    }

    fn grad(&self, ctx: &mut crate::gradient::GradientContext<T>) {
        let gy = ctx.output_grad();
        // use gy's array with rc increment.
        ctx.set_input_grads(vec![Some(gy)]);
    }
}

impl<T: Float> op::Op<T> for ELU<T> {
    fn name(&self) -> &str {
        "ELU"
    }

    fn compute(&self, ctx: &mut crate::runtime::OpComputeContext<T>) {
        let ret = ctx.input(0).mapv(move |a| {
            if a > T::zero() {
                a
            } else {
                self.alpha * (a.exp() - T::one())
            }
        });
        let ret = Ok(crate::ArrRepr::Owned(ret));
        ctx.push_output(ret)
    }

    fn grad(&self, ctx: &mut crate::gradient::GradientContext<T>) {
        let gy = &ctx.output_grad();
        let gx = Tensor::builder()
            .set_inputs(&[&ctx.input(0), gy])
            .set_shape(&ctx.scope().shape(gy))
            .build(ctx.scope(), ELUGrad { alpha: self.alpha });
        ctx.set_input_grads(vec![Some(gx)]);
    }
}

impl<T: Float> op::Op<T> for ELUGrad<T> {
    fn name(&self) -> &str {
        "ELUGrad"
    }

    fn compute(&self, ctx: &mut crate::runtime::OpComputeContext<T>) {
        let x = &ctx.input(0);
        let a = x.mapv(move |a| {
            if a > T::zero() {
                T::one()
            } else {
                self.alpha * (a.exp() - T::one()) + self.alpha
            }
        });
        let ret = Ok(crate::ArrRepr::Owned(a * &ctx.input(1)));
        ctx.push_output(ret)
    }

    fn grad(&self, ctx: &mut crate::gradient::GradientContext<T>) {
        ctx.set_input_grads(vec![None, None]);
    }
}
