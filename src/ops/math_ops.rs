use crate::ndarray_ext::{NdArray, NdArrayView};
use crate::Context;
use crate::op;
use crate::ops;
use crate::tensor::Tensor;
use crate::Float;
use ndarray;
use ndarray::Zip;

pub struct Sin;
pub struct Cos;
pub struct Tan;
pub struct Asin;
pub struct Acos;
pub struct Atan;
pub struct Sinh;
pub struct Cosh;
pub struct Tanh;
pub struct Asinh;
pub struct Acosh;
pub struct Atanh;
pub struct Exp;
pub struct Sqrt;
pub struct NegOp;
pub struct Floor;
pub struct Ceil;
pub struct Sign;
pub struct Reciprocal;
pub struct Square;
pub struct Abs;
pub struct Log<T: Float> {
    pub a: T,
}
pub struct Pow<T: Float> {
    pub a: T,
}
pub struct LogSumExp {
    pub axis: isize,
    pub keep_dims: bool,
}
pub struct Transpose {
    pub invert_axes: bool,
}

#[inline(always)]
fn equal<T: Float>(a: T, b: T) -> T {
    T::from((a == b) as i32).unwrap()
}
#[inline(always)]
fn not_equal<T: Float>(a: T, b: T) -> T {
    T::from((a != b) as i32).unwrap()
}
#[inline(always)]
fn greater<T: Float>(a: T, b: T) -> T {
    T::from((a > b) as i32).unwrap()
}
#[inline(always)]
fn lesser<T: Float>(a: T, b: T) -> T {
    T::from((a < b) as i32).unwrap()
}
#[inline(always)]
fn greater_equal<T: Float>(a: T, b: T) -> T {
    T::from((a >= b) as i32).unwrap()
}
#[inline(always)]
fn lesser_equal<T: Float>(a: T, b: T) -> T {
    T::from((a <= b) as i32).unwrap()
}
#[inline(always)]
fn maximum<T: Float>(a: T, b: T) -> T {
    a.max(b)
}
#[inline(always)]
fn minimum<T: Float>(a: T, b: T) -> T {
    a.min(b)
}

macro_rules! impl_cmp_op {
    ($struct_name:ident, $name:expr, $assign:expr, $grad_fn:expr) => {
        pub struct $struct_name;

        impl<'a, T: Float> op::Op<'a, T> for $struct_name {
            fn name(&self) -> &str {
                stringify!($struct_name)
            }

            fn compute(&self, ctx: &mut crate::runtime::OpComputeContext<T>) {
                let x0 = &ctx.input(0);
                let x1 = &ctx.input(1);
                let shape0 = x0.shape();
                let shape1 = x1.shape();

                let x0_is_scalar = crate::ndarray_ext::is_scalar_shape(shape0);
                let x1_is_scalar = crate::ndarray_ext::is_scalar_shape(shape1);

                let ret = if x0_is_scalar && x1_is_scalar {
                    let x1_elem = x1[ndarray::IxDyn(&[])];
                    x0.mapv(move |a| $assign(a, x1_elem))
                } else if x0_is_scalar && !x1_is_scalar {
                    let x0_elem = x0[ndarray::IxDyn(&[])];
                    x1.mapv(move |a| $assign(x0_elem, a))
                } else if !x0_is_scalar && x1_is_scalar {
                    let x1_elem = x1[ndarray::IxDyn(&[])];
                    x0.mapv(move |a| $assign(a, x1_elem))
                } else {
                    // case that scalar is not involved
                    // Check the input ranks.
                    // op couldn't we catch here cause ndarray's panics.

                    // rank check
                    if shape0.len() != shape1.len() {
                        panic!(
                            "Tensor ranks mismatch: {}({}'s lhs input) vs {}({}'s rhs input)",
                            shape0.len(),
                            $name,
                            shape1.len(),
                            $name,
                        )
                    }

                    let size0: usize = shape0.iter().product();
                    let size1: usize = shape1.iter().product();

                    // Whether broadcast of x0 and x1 is needed or not is depends on
                    // their shapes.
                    // FIXME: Is this cond branch ok?
                    if size0 < size1 {
                        let mut result = NdArray::zeros(shape1);
                        Zip::from(&mut result)
                            .and_broadcast(x0)
                            .and(x1)
                            .apply(|r, a, b| *r = $assign(a.clone(), b.clone()));
                        result
                    } else if size0 > size1 {
                        panic!(
                            "Tensor ranks mismatch: {}({}'s lhs input) vs {}({}'s rhs input)",
                            shape0.len(),
                            $name,
                            shape1.len(),
                            $name
                        );
                    } else {
                        // same
                        let mut result = NdArray::zeros(shape0);
                        Zip::from(&mut result)
                            .and(x0)
                            .and(x1)
                            .apply(|r, a, b| *r = $assign(a.clone(), b.clone()));
                        result
                    }
                };

                ctx.push_output(Ok(crate::ArrRepr::Owned(ret)));
            }

            fn grad(
                &self,
                gy: &'a Tensor<'a, T>,
                xs: &[&'a Tensor<'a, T>],
                y: &'a Tensor<'a, T>,
                c: &mut Context<'a, T>
            ) -> Vec<Option<&'a Tensor<'a, T>>> {
                $grad_fn(gy, xs, y, c)
            }
        }
    };
}

impl_cmp_op!(Equal, "Equal", equal, none_grad);
impl_cmp_op!(NotEqual, "NotEqual", not_equal, none_grad);
impl_cmp_op!(Greater, "Greater", greater, none_grad);
impl_cmp_op!(Lesser, "Lesser", lesser, none_grad);
impl_cmp_op!(GreaterEqual, "GreaterEqual", greater_equal, none_grad);
impl_cmp_op!(LesserEqual, "LesserEqual", lesser_equal, none_grad);
impl_cmp_op!(Maximum, "Maximum", maximum, min_max_grad);
impl_cmp_op!(Minimum, "Minimum", minimum, min_max_grad);

fn none_grad<'a, T: Float>(_: &'a Tensor<'a, T>, _: &[&'a Tensor<'a, T>], _: &'a Tensor<'a, T>, c: &mut Context<'a, T>) -> Vec<Option<&'a Tensor<'a, T>>> {
    vec![None]
}

fn min_max_grad<'a, T: Float>(
    gy: &'a Tensor<'a, T>,
    xs: &[&'a Tensor<'a, T>],
    y: &'a Tensor<'a, T>,
    c: &mut Context<'a, T>
) -> Vec<Option<&'a Tensor<'a, T>>> {
    let a = xs[0];
    let b = xs[1];
    let selected_a = c.equal(a, y);
    let selected_b = c.equal(b, y);
    vec![
        Some(c.mul(selected_a, gy)),
        Some(c.mul(selected_b, gy)),
    ]
}

impl<'a, T: Float> op::Op<'a, T> for Abs {
    fn name(&self) -> &str {
        "Abs"
    }

    fn compute(&self, ctx: &mut crate::runtime::OpComputeContext<T>) {
        let ret = ctx.input(0).map(|x| x.abs());
        ctx.push_output(Ok(crate::ArrRepr::Owned(ret)));
    }

    fn grad(&self, gy: &'a Tensor<'a, T>, inputs: &[&'a Tensor<'a, T>], _: &'a Tensor<'a, T>, c: &mut Context<'a, T>) -> Vec<Option<&'a Tensor<'a, T>>> {
        vec![Some(gy * c.sign(inputs[0]))]
    }
}

impl<'a, T: Float> op::Op<'a, T> for NegOp {
    fn name(&self) -> &str {
        "Neg"
    }

    fn compute(&self, ctx: &mut crate::runtime::OpComputeContext<T>) {
        let ret = ctx.input(0).map(|x| x.neg());
        ctx.push_output(Ok(crate::ArrRepr::Owned(ret)));
    }

    fn grad(&self, gy: &'a Tensor<'a, T>, _: &[&'a Tensor<'a, T>], _: &'a Tensor<'a, T>, c: &mut Context<'a, T>) -> Vec<Option<&'a Tensor<'a, T>>> {
        vec![Some(c.neg(gy))]
    }
}

impl<'a, T: Float> op::Op<'a, T> for Square {
    fn name(&self) -> &str {
        "Square"
    }

    fn compute(&self, ctx: &mut crate::runtime::OpComputeContext<T>) {
        let ret = ctx.input(0).map(|&x| x * x);
        ctx.push_output(Ok(crate::ArrRepr::Owned(ret)));
    }

    fn grad(&self, gy: &'a Tensor<'a, T>, inputs: &[&'a Tensor<'a, T>], _: &'a Tensor<'a, T>, c: &mut Context<'a, T>) -> Vec<Option<&'a Tensor<'a, T>>> {
        let two = c.scalar(T::one() + T::one());
        vec![Some(two * inputs[0] * gy)]
    }
}

impl<'a, T: Float> op::Op<'a, T> for Reciprocal {
    fn name(&self) -> &str {
        "Reciprocal"
    }

    fn compute(&self, ctx: &mut crate::runtime::OpComputeContext<T>) {
        let ret = ctx.input(0).map(|x| x.recip());
        ctx.push_output(Ok(crate::ArrRepr::Owned(ret)));
    }

    fn grad(&self, gy: &'a Tensor<'a, T>, _: &[&'a Tensor<'a, T>], output: &'a Tensor<'a, T>, c: &mut Context<'a, T>) -> Vec<Option<&'a Tensor<'a, T>>> {
        vec![Some(c.neg(&c.square(output)) * gy)]
    }
}

impl<'a, T: Float> op::Op<'a, T> for Sign {
    fn name(&self) -> &str {
        "Sign"
    }

    fn compute(&self, ctx: &mut crate::runtime::OpComputeContext<T>) {
        let ret = ctx.input(0).mapv(|x| {
            if x == T::zero() {
                T::zero()
            } else {
                x.signum()
            }
        });
        ctx.push_output(Ok(crate::ArrRepr::Owned(ret)));
    }

    fn grad(&self, _: &'a Tensor<'a, T>, _: &[&'a Tensor<'a, T>], _: &'a Tensor<'a, T>, c: &mut Context<'a, T>) -> Vec<Option<&'a Tensor<'a, T>>> {
        vec![None]
    }
}

impl<'a, T: Float> op::Op<'a, T> for Floor {
    fn name(&self) -> &str {
        "Floor"
    }

    fn compute(&self, ctx: &mut crate::runtime::OpComputeContext<T>) {
        let ret = ctx.input(0).map(|x| x.floor());
        ctx.push_output(Ok(crate::ArrRepr::Owned(ret)));
    }

    fn grad(&self, _: &'a Tensor<'a, T>, _: &[&'a Tensor<'a, T>], _: &'a Tensor<'a, T>, c: &mut Context<'a, T>) -> Vec<Option<&'a Tensor<'a, T>>> {
        vec![None]
    }
}

impl<'a, T: Float> op::Op<'a, T> for Ceil {
    fn name(&self) -> &str {
        "Ceil"
    }

    fn compute(&self, ctx: &mut crate::runtime::OpComputeContext<T>) {
        let ret = ctx.input(0).map(|x| x.ceil());
        ctx.push_output(Ok(crate::ArrRepr::Owned(ret)));
    }

    fn grad(&self, _: &'a Tensor<'a, T>, _: &[&'a Tensor<'a, T>], _: &'a Tensor<'a, T>, c: &mut Context<'a, T>) -> Vec<Option<&'a Tensor<'a, T>>> {
        vec![None]
    }
}

impl<'a, T: Float> op::Op<'a, T> for Transpose {
    fn name(&self) -> &str {
        "Transpose"
    }

    fn compute(&self, ctx: &mut crate::runtime::OpComputeContext<T>) {
        let perm = &ctx.input(1);
        let perm_len = perm.len();
        assert!(perm_len >= 2);

        let ret = unsafe {
            let mut dims = crate::uninitialized_vec::<usize>(perm_len);
            for (i, d) in perm.iter().enumerate() {
                if self.invert_axes {
                    dims[d.to_usize().unwrap()] = i;
                } else {
                    dims[i] = d.to_usize().unwrap();
                }
            }
            ctx.input(0).permuted_axes(dims.as_slice())
        };

        ctx.push_output(Ok(crate::ArrRepr::View(ret)));
    }

    fn grad(&self, gy: &'a Tensor<'a, T>, inputs: &[&'a Tensor<'a, T>], _: &'a Tensor<'a, T>, c: &mut Context<'a, T>) -> Vec<Option<&'a Tensor<'a, T>>> {
        let gx = Tensor::builder()
            .set_inputs(&[gy, inputs[1]])
            .set_shape(c.shape(inputs[0]))
            .build(c, Transpose {
                invert_axes: !self.invert_axes,
            });
        vec![Some(gx), None]
    }
}

pub fn logsumexp_forward<T: Float>(x: &NdArrayView<T>, axis: isize, keep_dims: bool) -> NdArray<T> {
    let axis = if axis < 0 {
        (x.ndim() as isize + axis) as usize
    } else {
        axis as usize
    };

    let mut a = x.shape().to_vec();
    if keep_dims {
        a[axis] = 1;
    } else {
        a.remove(axis);
    }
    let reduced_shape = a.as_slice();

    let max_fn = T::max;
    let min_val = T::min_value();
    let ref max = x
        .fold_axis(ndarray::Axis(axis), min_val, move |&a, &b| max_fn(a, b))
        .into_shape(ndarray::IxDyn(reduced_shape))
        .unwrap();

    let exp = {
        // subtract `max` to prevent overflow of exp
        let mut tmp = x - max;
        tmp.mapv_inplace(|a| a.exp());
        tmp
    };

    // unwrap is safe
    let mut sum = exp
        .sum_axis(ndarray::Axis(axis))
        .into_shape(ndarray::IxDyn(reduced_shape))
        .unwrap();

    use std::f64;
    let e = T::from(f64::consts::E).unwrap();
    sum.mapv_inplace(move |a| a.log(e));
    sum += max;
    sum
}

impl<'a, T: Float> op::Op<'a, T> for LogSumExp {
    fn name(&self) -> &str {
        "LogSumExp"
    }

    fn compute(&self, ctx: &mut crate::runtime::OpComputeContext<T>) {
        let ret = logsumexp_forward(&ctx.input(0), self.axis, self.keep_dims);
        ctx.push_output(Ok(crate::ArrRepr::Owned(ret)));
    }

    fn grad(&self, gy: &'a Tensor<'a, T>, inputs: &[&'a Tensor<'a, T>], _: &'a Tensor<'a, T>, c: &mut Context<'a, T>) -> Vec<Option<&'a Tensor<'a, T>>> {
        // let ref sum = c.exp(output);
        // let ref exp = c.exp(inputs[0]);
        // let gx = gy * exp / sum;
        let gx = c.softmax(inputs[0], self.axis) * gy;
        vec![Some(gx)]
    }
}

impl<'a, T: Float> op::Op<'a, T> for Pow<T> {
    fn name(&self) -> &str {
        "Pow"
    }

    fn compute(&self, ctx: &mut crate::runtime::OpComputeContext<T>) {
        let a = self.a;
        let ret = ctx.input(0).map(move |x| x.powf(a));
        ctx.push_output(Ok(crate::ArrRepr::Owned(ret)));
    }

    fn grad(&self, gy: &'a Tensor<'a, T>, inputs: &[&'a Tensor<'a, T>], _: &'a Tensor<'a, T>, c: &mut Context<'a, T>) -> Vec<Option<&'a Tensor<'a, T>>> {
        let x = inputs[0];
        let gx = gy * self.a * c.pow(x, self.a - T::one());
        vec![Some(gx)]
    }
}

impl<'a, T: Float> op::Op<'a, T> for Sqrt {
    fn name(&self) -> &str {
        "Sqrt"
    }

    fn compute(&self, ctx: &mut crate::runtime::OpComputeContext<T>) {
        let ret = ctx.input(0).map(|a| a.sqrt());
        ctx.push_output(Ok(crate::ArrRepr::Owned(ret)));
    }

    fn grad(&self, gy: &'a Tensor<'a, T>, inputs: &[&'a Tensor<'a, T>], _: &'a Tensor<'a, T>, c: &mut Context<'a, T>) -> Vec<Option<&'a Tensor<'a, T>>> {
        let x = inputs[0];
        let half = c.scalar(T::one());
        let ret = half * c.pow(x, T::one().neg());
        vec![Some(gy * ret)]
    }
}

impl<'a, T: Float> op::Op<'a, T> for Log<T> {
    fn name(&self) -> &str {
        "Log"
    }

    fn compute(&self, ctx: &mut crate::runtime::OpComputeContext<T>) {
        let ret = ctx.input(0).map(move |a| a.log(self.a));
        ctx.push_output(Ok(crate::ArrRepr::Owned(ret)));
    }

    fn grad(&self, gy: &'a Tensor<'a, T>, inputs: &[&'a Tensor<'a, T>], _: &'a Tensor<'a, T>, c: &mut Context<'a, T>) -> Vec<Option<&'a Tensor<'a, T>>> {
        vec![Some(gy / inputs[0])]
    }
}

impl<'a, T: Float> op::Op<'a, T> for Exp {
    fn name(&self) -> &str {
        "Exp"
    }

    fn compute(&self, ctx: &mut crate::runtime::OpComputeContext<T>) {
        let ret = ctx.input(0).map(|a| a.exp());
        ctx.push_output(Ok(crate::ArrRepr::Owned(ret)));
    }

    fn grad(&self, gy: &'a Tensor<'a, T>, _: &[&'a Tensor<'a, T>], output: &'a Tensor<'a, T>, c: &mut Context<'a, T>) -> Vec<Option<&'a Tensor<'a, T>>> {
        vec![Some(output * gy)]
    }
}

impl<'a, T: Float> op::Op<'a, T> for Atanh {
    fn name(&self) -> &str {
        "Atanh"
    }

    fn compute(&self, ctx: &mut crate::runtime::OpComputeContext<T>) {
        let ret = ctx.input(0).map(|a| a.atanh());
        ctx.push_output(Ok(crate::ArrRepr::Owned(ret)));
    }

    fn grad(&self, gy: &'a Tensor<'a, T>, inputs: &[&'a Tensor<'a, T>], _: &'a Tensor<'a, T>, c: &mut Context<'a, T>) -> Vec<Option<&'a Tensor<'a, T>>> {
        let x = inputs[0];
        let one = c.scalar(T::one());
        let y = c.reciprocal(one - c.square(x));
        vec![Some(y * gy)]
    }
}

impl<'a, T: Float> op::Op<'a, T> for Acosh {
    fn name(&self) -> &str {
        "Acosh"
    }

    fn compute(&self, ctx: &mut crate::runtime::OpComputeContext<T>) {
        let ret = ctx.input(0).map(|a| a.acosh());
        ctx.push_output(Ok(crate::ArrRepr::Owned(ret)));
    }

    fn grad(&self, gy: &'a Tensor<'a, T>, inputs: &[&'a Tensor<'a, T>], _: &'a Tensor<'a, T>, c: &mut Context<'a, T>) -> Vec<Option<&'a Tensor<'a, T>>> {
        let x = inputs[0];
        let one = c.scalar(T::one().neg());
        let y = one / c.sqrt(c.square(x) + one);
        vec![Some(y * gy)]
    }
}

impl<'a, T: Float> op::Op<'a, T> for Asinh {
    fn name(&self) -> &str {
        "Asinh"
    }

    fn compute(&self, ctx: &mut crate::runtime::OpComputeContext<T>) {
        let ret = ctx.input(0).map(|a| a.asinh());
        ctx.push_output(Ok(crate::ArrRepr::Owned(ret)));
    }

    fn grad(&self, gy: &'a Tensor<'a, T>, inputs: &[&'a Tensor<'a, T>], _: &'a Tensor<'a, T>, c: &mut Context<'a, T>) -> Vec<Option<&'a Tensor<'a, T>>> {
        let x = inputs[0];
        let one = c.scalar(T::one());
        let y = one / c.sqrt(x * x + one);
        vec![Some(y * gy)]
    }
}

impl<'a, T: Float> op::Op<'a, T> for Tanh {
    fn name(&self) -> &str {
        "Tanh"
    }

    fn compute(&self, ctx: &mut crate::runtime::OpComputeContext<T>) {
        let ret = ctx.input(0).map(|a| a.tanh());
        ctx.push_output(Ok(crate::ArrRepr::Owned(ret)));
    }

    fn grad(&self, gy: &'a Tensor<'a, T>, _: &[&'a Tensor<'a, T>], y: &'a Tensor<'a, T>, c: &mut Context<'a, T>) -> Vec<Option<&'a Tensor<'a, T>>> {
        vec![Some(gy * (c.scalar(T::one()) - c.square(y)))]
    }
}

impl<'a, T: Float> op::Op<'a, T> for Cosh {
    fn name(&self) -> &str {
        "Cosh"
    }

    fn compute(&self, ctx: &mut crate::runtime::OpComputeContext<T>) {
        let ret = ctx.input(0).map(|a| a.cosh());
        ctx.push_output(Ok(crate::ArrRepr::Owned(ret)));
    }

    fn grad(&self, gy: &'a Tensor<'a, T>, inputs: &[&'a Tensor<'a, T>], _: &'a Tensor<'a, T>, c: &mut Context<'a, T>) -> Vec<Option<&'a Tensor<'a, T>>> {
        vec![Some(c.sinh(inputs[0]) * gy)]
    }
}

impl<'a, T: Float> op::Op<'a, T> for Sinh {
    fn name(&self) -> &str {
        "Sinh"
    }

    fn compute(&self, ctx: &mut crate::runtime::OpComputeContext<T>) {
        let ret = ctx.input(0).map(|a| a.sinh());
        ctx.push_output(Ok(crate::ArrRepr::Owned(ret)));
    }

    fn grad(&self, gy: &'a Tensor<'a, T>, inputs: &[&'a Tensor<'a, T>], _: &'a Tensor<'a, T>, c: &mut Context<'a, T>) -> Vec<Option<&'a Tensor<'a, T>>> {
        vec![Some(c.cosh(inputs[0]) * gy)]
    }
}

impl<'a, T: Float> op::Op<'a, T> for Atan {
    fn name(&self) -> &str {
        "Atan"
    }

    fn compute(&self, ctx: &mut crate::runtime::OpComputeContext<T>) {
        let ret = ctx.input(0).map(|a| a.atan());
        ctx.push_output(Ok(crate::ArrRepr::Owned(ret)));
    }

    fn grad(&self, gy: &'a Tensor<'a, T>, inputs: &[&'a Tensor<'a, T>], _: &'a Tensor<'a, T>, c: &mut Context<'a, T>) -> Vec<Option<&'a Tensor<'a, T>>> {
        let x = inputs[0];
        let y = c.reciprocal(c.square(x) + T::one());
        vec![Some(y * gy)]
    }
}

impl<'a, T: Float> op::Op<'a, T> for Acos {
    fn name(&self) -> &str {
        "Acos"
    }

    fn compute(&self, ctx: &mut crate::runtime::OpComputeContext<T>) {
        let ret = ctx.input(0).map(|a| a.acos());
        ctx.push_output(Ok(crate::ArrRepr::Owned(ret)));
    }

    fn grad(&self, gy: &'a Tensor<'a, T>, inputs: &[&'a Tensor<'a, T>], _: &'a Tensor<'a, T>, c: &mut Context<'a, T>) -> Vec<Option<&'a Tensor<'a, T>>> {
        let x = inputs[0];
        let y = c.scalar(T::one().neg()) / c.sqrt(c.scalar(T::one()) - c.square(x));
        vec![Some(y * gy)]
    }
}

impl<'a, T: Float> op::Op<'a, T> for Asin {
    fn name(&self) -> &str {
        "Asin"
    }

    fn compute(&self, ctx: &mut crate::runtime::OpComputeContext<T>) {
        let ret = ctx.input(0).map(|a| a.asin());
        ctx.push_output(Ok(crate::ArrRepr::Owned(ret)));
    }

    fn grad(&self, gy: &'a Tensor<'a, T>, inputs: &[&'a Tensor<'a, T>], _: &'a Tensor<'a, T>, c: &mut Context<'a, T>) -> Vec<Option<&'a Tensor<'a, T>>> {
        let x = inputs[0];
        let y = c.scalar(T::one()) / c.sqrt(c.scalar(T::one()) - x * x);
        vec![Some(y * gy)]
    }
}

impl<'a, T: Float> op::Op<'a, T> for Sin {
    fn name(&self) -> &str {
        "Sin"
    }

    fn compute(&self, ctx: &mut crate::runtime::OpComputeContext<T>) {
        let ret = ctx.input(0).map(|a| a.sin());
        ctx.push_output(Ok(crate::ArrRepr::Owned(ret)));
    }

    fn grad(&self, gy: &'a Tensor<'a, T>, inputs: &[&'a Tensor<'a, T>], _: &'a Tensor<'a, T>, c: &mut Context<'a, T>) -> Vec<Option<&'a Tensor<'a, T>>> {
        vec![Some(c.cos(inputs[0]) * gy)]
    }
}

impl<'a, T: Float> op::Op<'a, T> for Cos {
    fn name(&self) -> &str {
        "Cos"
    }

    fn compute(&self, ctx: &mut crate::runtime::OpComputeContext<T>) {
        let ret = ctx.input(0).map(|a| a.cos());
        ctx.push_output(Ok(crate::ArrRepr::Owned(ret)));
    }

    fn grad(&self, gy: &'a Tensor<'a, T>, inputs: &[&'a Tensor<'a, T>], _: &'a Tensor<'a, T>, c: &mut Context<'a, T>) -> Vec<Option<&'a Tensor<'a, T>>> {
        vec![Some(c.neg(&(c.sin(inputs[0]) * gy)))]
    }
}

impl<'a, T: Float> op::Op<'a, T> for Tan {
    fn name(&self) -> &str {
        "Tan"
    }

    fn compute(&self, ctx: &mut crate::runtime::OpComputeContext<T>) {
        let ret = ctx.input(0).map(|a| a.tan());
        ctx.push_output(Ok(crate::ArrRepr::Owned(ret)));
    }

    fn grad(&self, gy: &'a Tensor<'a, T>, inputs: &[&'a Tensor<'a, T>], _: &'a Tensor<'a, T>, c: &mut Context<'a, T>) -> Vec<Option<&'a Tensor<'a, T>>> {
        let ref cos = c.cos(inputs[0]);
        vec![Some(gy / c.square(cos))]
    }
}
