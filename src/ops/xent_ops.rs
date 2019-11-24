use crate::ndarray_ext::{NdArray, NdArrayView};
use crate::op;
use crate::ops;
use crate::tensor::Tensor;
use crate::Float;
use ndarray;

pub struct SoftmaxCrossEntropy;
pub struct SparseSoftmaxCrossEntropy;
pub struct SparseSoftmaxCrossEntropyGrad;
pub struct SigmoidCrossEntropy;
pub struct LogSoftmax {
    pub axis: isize,
}

impl<T: Float> op::Op<T> for LogSoftmax {
    fn name(&self) -> &str {
        "LogSoftmax"
    }

    fn compute(&self, ctx: &mut crate::runtime::OpComputeContext<T>) {
        let x = ctx.input(0);
        ctx.push_output(Ok(crate::ArrRepr::Owned(
            (&x) - &ops::math_ops::logsumexp_forward(&x, self.axis, true),
        )));
    }

    fn grad(&self, gy: &Tensor<T>, _: &[&Tensor<T>], output: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        let sm = ops::exp(output);
        let sum = ops::reduce_sum(gy, &[1], true);
        let ref mul = sm * sum;
        vec![Some(gy - mul)]
    }
}

impl<T: Float> op::Op<T> for SigmoidCrossEntropy {
    fn name(&self) -> &str {
        "SigmoidCrossEntropy"
    }

    fn compute(&self, ctx: &mut crate::runtime::OpComputeContext<T>) {
        let x: &NdArrayView<T> = &ctx.input(0);
        let t: &NdArrayView<T> = &ctx.input(1);

        assert_eq!(x.shape(), t.shape(), "x.shape must match t.shape");

        use std::f64;
        let e = T::from(f64::consts::E).unwrap();
        let max_fn = T::max;
        let mut tmp: NdArray<T> =
            x.mapv(move |a| ((-a.abs()).exp() + T::one()).log(e) + max_fn(T::zero(), a));
        tmp -= &(t * x);
        ctx.push_output(Ok(crate::ArrRepr::Owned(tmp)));
    }

    fn grad(&self, gy: &Tensor<T>, inputs: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        let x = inputs[0];
        let t = inputs[1];

        let gx1 = {
            let ref exp = ops::exp(x);
            ((exp / (ops::scalar(T::one()) + exp)) - t) * gy
        };

        let gx2 = ops::neg(&(gy * t));

        vec![Some(gx1), Some(gx2)]
    }
}

impl<T: Float> op::Op<T> for SparseSoftmaxCrossEntropy {
    fn name(&self) -> &str {
        "SparseSoftmaxCrossEntropy"
    }

    fn compute(&self, ctx: &mut crate::runtime::OpComputeContext<T>) {
        let (x, t) = (&ctx.input(0), &ctx.input(1));
        let log_x: NdArray<T> = x - &ops::math_ops::logsumexp_forward(x, 1, true);

        // validation
        {
            let t_shape = t.shape();
            assert_eq!(log_x.ndim(), 2, "Bad first argument's shape");
            let t_rank = t_shape.len();
            if t_rank == 2 {
                assert_eq!(t_shape[1], 1, "Bad second argument's shape");
            } else if t_rank != 1 {
                panic!("Bad second argument's shape");
            }
        }

        let mut t_iter = t.iter();
        let ret = log_x
            .map_axis(ndarray::Axis(1), move |row| {
                -row[t_iter.next().unwrap().to_usize().unwrap()]
            })
            .into_shape(ndarray::IxDyn(&[log_x.shape()[0], 1]))
            .unwrap();

        ctx.push_output(Ok(crate::ArrRepr::Owned(ret)));
        ctx.push_output(Ok(crate::ArrRepr::Owned(log_x)));
    }

    fn grad(
        &self,
        gy: &Tensor<T>,
        inputs: &[&Tensor<T>],
        output: &Tensor<T>,
    ) -> Vec<Option<Tensor<T>>> {
        let t = inputs[1];
        let ref log_x = ops::nth_tensor(output, 1);

        let gx1 = Tensor::builder()
            .set_inputs(&[log_x, t, gy])
            .build(SparseSoftmaxCrossEntropyGrad);

        // gx2 won't be used in most cases.
        let gx2 = {
            let ref x = ops::exp(log_x);
            let sum = ops::reduce_sum(&(x * log_x), &[1], true);
            x * gy * (sum - log_x)
        };

        vec![Some(gx1), Some(gx2)]
    }
}

impl<T: Float> op::Op<T> for SparseSoftmaxCrossEntropyGrad {
    fn name(&self) -> &str {
        "SparseSoftmaxCrossEntropyGrad"
    }

    fn compute(&self, ctx: &mut crate::runtime::OpComputeContext<T>) {
        let log_x = &ctx.input(0); // x is softmax
        let mut x = log_x.map(|a| a.exp());
        let t = &ctx.input(1);
        for (mut row, &t_) in x.axis_iter_mut(ndarray::Axis(0)).zip(t) {
            row[t_.to_usize().unwrap()] -= T::one();
        }

        let gy = &ctx.input(2);
        x *= gy;
        ctx.push_output(Ok(crate::ArrRepr::Owned(x)));
    }

    fn grad(&self, _: &Tensor<T>, _: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![None, None]
    }
}

impl<T: Float> op::Op<T> for SoftmaxCrossEntropy {
    fn name(&self) -> &str {
        "SoftmaxCrossEntropy"
    }

    fn compute(&self, ctx: &mut crate::runtime::OpComputeContext<T>) {
        let x = &ctx.input(0);
        let log_x: NdArray<T> = x - &ops::math_ops::logsumexp_forward(x, 1, true);
        // `t` must be one-hot
        let t = &ctx.input(1);
        assert_eq!(log_x.ndim(), 2, "x must be 2-ranked tensor");
        assert_eq!(t.ndim(), 2, "t must be 2-ranked tensor");
        // - t log x ( =(batch, num_classes))
        let minus_one = T::one().neg();
        ctx.push_output( Ok(crate::ArrRepr::Owned(
            (t * &log_x)
                .sum_axis(ndarray::Axis(1))
                .mapv(move |elem| elem * minus_one),
        )));
        ctx.push_output(Ok(crate::ArrRepr::Owned(log_x)));
    }

    fn grad(
        &self,
        gy: &Tensor<T>,
        inputs: &[&Tensor<T>],
        output: &Tensor<T>,
    ) -> Vec<Option<Tensor<T>>> {
        let ref log_x = ops::nth_tensor(output, 1);
        let ref x = ops::exp(log_x);
        let t = inputs[1];

        // x = softmax, gy = dy/dx
        // = {gy - Σ(x * gy)} * x
        // = {-t/x - Σ(x * -t/x)} * x
        // = {-t/x + Σt} * x
        // = -t + x
        let gx1 = (x - t) * gy;

        // gx2 won't be used in most cases
        let gx2 = {
            let sum = ops::reduce_sum(&(x * log_x), &[-1], true);
            gy * (sum - log_x) * output
        };

        vec![Some(gx1), Some(gx2)]
    }
}
