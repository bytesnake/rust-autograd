use crate::ndarray_ext;
use crate::ndarray_ext::NdArray;
use crate::op;
use crate::tensor::Tensor;
use crate::Float;
use ndarray;

pub struct Zeros;
pub struct Ones;
pub struct Range;
pub struct ConvertToTensor<T: Float> {
    pub arr: NdArray<T>,
}
pub struct Scalar<T: Float> {
    pub val: T,
}

impl<T: Float> op::Op<T> for Scalar<T> {
    fn name(&self) -> &str {
        "Scalar"
    }

    fn compute(&self, ctx: &mut crate::runtime::OpComputeContext<T>) {
        ctx.set_output(vec![Ok(crate::ArrRepr::Owned(
            ndarray::arr0(self.val).into_dyn(),
        ))]);
    }

    fn grad(&self, _: &Tensor<T>, _: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![None]
    }
}

impl<T: Float> op::Op<T> for Zeros {
    fn name(&self) -> &str {
        "Zeros"
    }

    fn compute(&self, ctx: &mut crate::runtime::OpComputeContext<T>) {
        let shape = &ctx.input(0);
        let ret = if let Some(a) = shape.as_slice() {
            ndarray_ext::zeros(
                a.iter()
                    .map(|&b| b.to_usize().unwrap())
                    .collect::<Vec<_>>()
                    .as_slice(),
            )
        } else {
            ndarray_ext::zeros(
                shape
                    .iter()
                    .map(|&b| b.to_usize().unwrap())
                    .collect::<Vec<_>>()
                    .as_slice(),
            )
        };
        ctx.set_output(vec![Ok(crate::ArrRepr::Owned(ret))]);
    }

    fn grad(&self, _: &Tensor<T>, _: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![None]
    }
}

impl<T: Float> op::Op<T> for Ones {
    fn name(&self) -> &str {
        "Ones"
    }

    fn compute(&self, ctx: &mut crate::runtime::OpComputeContext<T>) {
        let shape = &ctx.input(0);
        let ret = if let Some(a) = shape.as_slice() {
            ndarray_ext::ones(
                a.iter()
                    .map(|&b| b.to_usize().unwrap())
                    .collect::<Vec<_>>()
                    .as_slice(),
            )
        } else {
            ndarray_ext::ones(
                shape
                    .iter()
                    .map(|&b| b.to_usize().unwrap())
                    .collect::<Vec<_>>()
                    .as_slice(),
            )
        };
        ctx.set_output(vec![Ok(crate::ArrRepr::Owned(ret))]);
    }

    fn grad(&self, _: &Tensor<T>, _: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![None]
    }
}

impl<T: Float> op::Op<T> for Range {
    fn name(&self) -> &str {
        "Range"
    }

    fn compute(&self, ctx: &mut crate::runtime::OpComputeContext<T>) {
        let x0 = &ctx.input(0);
        let x1 = &ctx.input(1);
        let x2 = &ctx.input(2);

        let true_shape = &[];
        if x0.shape() != true_shape || x1.shape() != true_shape || x2.shape() != true_shape {
            panic!("Inputs to `range` should be 0-ranked tensors");
        }

        let start = x0[ndarray::IxDyn(&[])];
        let end = x1[ndarray::IxDyn(&[])];
        let step = x2[ndarray::IxDyn(&[])];
        assert!(start < end, "`start` and `end` overlap.");
        ctx.set_output(vec![Ok(crate::ArrRepr::Owned(
            ndarray::Array1::range(start, end, step).into_dyn(),
        ))]);
    }

    fn grad(&self, _: &Tensor<T>, _: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![None, None, None]
    }
}

impl<T: Float> op::Op<T> for ConvertToTensor<T> {
    fn name(&self) -> &str {
        "ConvertToTensor"
    }

    fn compute(&self, ctx: &mut crate::runtime::OpComputeContext<T>) {
        ctx.set_output(vec![Ok(crate::ArrRepr::Owned(self.arr.clone()))]);
    }

    fn grad(&self, _: &Tensor<T>, _: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![]
    }
}
