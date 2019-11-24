use crate::ndarray_ext::{NdArray, NdArrayView};
use crate::op;
use crate::ops;
use crate::tensor::Tensor;
use crate::Float;
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

    fn grad(&self, gy: &Tensor<T>, _: &[&Tensor<T>], output: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        let sum = ops::reduce_sum(&(output * gy), &[self.axis], true);
        vec![Some((gy - sum) * output)]
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

    fn grad(&self, gy: &Tensor<T>, xs: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        let a = &ops::exp(xs[0]);
        let b = a + ops::scalar(T::one());
        let gx = gy * (a / b);
        vec![Some(gx)]
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

    fn grad(&self, gy: &Tensor<T>, _: &[&Tensor<T>], y: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![Some(gy * (y - ops::square(y)))]
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

    fn grad(&self, gy: &Tensor<T>, inputs: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        let bin = ops::greater(inputs[0], &ops::scalar(T::zero()));
        vec![Some(ops::mul(bin, gy))]
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

    fn grad(&self, gy: &Tensor<T>, _: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        // use gy's array with rc increment.
        vec![Some(gy.clone())]
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

    fn grad(&self, gy: &Tensor<T>, inputs: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        let gx = Tensor::builder()
            .set_inputs(&[inputs[0], gy])
            .set_shape(gy.shape())
            .build(ELUGrad { alpha: self.alpha });
        vec![Some(gx)]
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

    fn grad(&self, _: &Tensor<T>, _: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![None, None]
    }
}
