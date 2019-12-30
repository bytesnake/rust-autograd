use super::*;

pub struct Conv2DTranspose {
    pub pad: usize,
    pub stride: usize,
    pub dilation: usize,
}

pub struct Conv2DTransposeFilterGrad {
    pub pad: usize,
    pub stride: usize,
    pub dilation: usize,
}

impl<T: Float> crate::op::Op<T> for Conv2DTranspose {
    #[allow(unused_mut)]
    fn compute(&self, ctx: &mut crate::op::OpComputeContext<T>) {
        let gy = &ctx.input(0); // (batch, ych, yh, yw)
        let w = &ctx.input(1); // (ych, xch, kh, kw)
        let gy_shape = gy.shape();
        let f_shape = w.shape();

        let batch_size = gy_shape[0];
        let ych = gy_shape[1];
        let yh = gy_shape[2];
        let yw = gy_shape[3];

        let xch = f_shape[1];
        let kh = f_shape[2];
        let kw = f_shape[3];
        let xh = get_xh!(self, yh, kh);
        let xw = get_xw!(self, yw, kw);

        assert_eq!(
            gy_shape.len(),
            4,
            "ag::conv2d: Input must be 4D (got {:?})",
            gy_shape
        );
        assert_eq!(
            f_shape.len(),
            4,
            "ag::conv2d: Filter must be 4D (got {:?})",
            f_shape
        );
        assert_eq!(
            ych, f_shape[0],
            "ag::conv2d: Number of input channels ({:?}) must match second filter dim ({:?})",
            ych, f_shape[0]
        );

        // sgemm params
        let k = ych;
        let n = yh * yw;
        let m = kh * kw * xch;

        let size_per_batch_col = xch * kh * kw * yh * yw;

        let copied_gy = ndarray_ext::copy_if_dirty(&gy);
        let copied_w = ndarray_ext::copy_if_dirty(w);
        let gy_ptr = copied_gy.map(|inner| inner.as_ptr()).unwrap_or(gy.as_ptr());
        let w_ptr = copied_w.map(|inner| inner.as_ptr()).unwrap_or(w.as_ptr());

        let gx = unsafe {
            let mut col = uninitialized_vec(batch_size * size_per_batch_col);

            #[cfg(feature = "mkl")]
            {
                if same_type::<T, f32>() {
                    crate::ops::dot_ops::cblas_sgemm_batch_wrapper(
                        true,
                        false,
                        m,
                        n,
                        k,
                        &[1.],
                        vec![w_ptr as *const f32; batch_size],
                        crate::dot_ops::get_region_heads(batch_size, gy_ptr, gy.len()),
                        &[0.],
                        crate::dot_ops::get_region_heads(batch_size, col.as_ptr(), col.len()),
                        1,
                        batch_size,
                    );
                } else if same_type::<T, f64>() {
                    crate::ops::dot_ops::cblas_dgemm_batch_wrapper(
                        true,
                        false,
                        m,
                        n,
                        k,
                        &[1.],
                        vec![w_ptr as *const f64; batch_size],
                        crate::dot_ops::get_region_heads(batch_size, gy_ptr, gy.len()),
                        &[0.],
                        crate::dot_ops::get_region_heads(batch_size, col.as_ptr(), col.len()),
                        1,
                        batch_size,
                    );
                } else {
                    panic!("gemm supports only f32 and f64.")
                }
            }
            #[cfg(not(feature = "mkl"))]
            {
                let w_slice = slice::from_raw_parts(w_ptr, w.len());
                let gy_slice = slice::from_raw_parts(gy_ptr, gy.len());
                let col_ref = &col[0];
                let size_per_batch_gy = ych * yh * yw;
                (0..batch_size).into_par_iter().for_each(|i| {
                    let w = w_slice.as_ptr();
                    let gy_region_head = gy_slice.as_ptr().offset((i * size_per_batch_gy) as isize);
                    let col_region_head: *mut T = mem::transmute(col_ref);
                    let col_region_head = col_region_head.offset((i * size_per_batch_col) as isize);
                    slow_gemm!(
                        true,
                        false,
                        w as *const f32,
                        gy_region_head as *const f32,
                        col_region_head as *mut f32,
                        m,
                        n,
                        k,
                        1.,
                        0.
                    );
                });
            }

            col2im_batch(
                col.as_slice(),
                batch_size,
                xch as i32,
                xh as i32,
                xw as i32,
                kh as i32,
                kw as i32,
                self.pad as i32,
                self.pad as i32,
                self.stride as i32,
                self.stride as i32,
                self.dilation as i32,
                self.dilation as i32,
            )
        };
        let gx = NdArray::from_shape_vec(ndarray::IxDyn(&[batch_size, xch, xh, xw]), gx);
        ctx.push_output(Ok(crate::ArrRepr::Owned(gx.unwrap())));
    }

    fn grad(&self, ctx: &mut crate::op::OpGradientContext<T>) {
        let s = ctx.graph();
        let x = ctx.input(0);
        let w = ctx.input(1);
        let gy = ctx.output_grad();

        let gx = Tensor::builder().set_inputs(&[&gy, &w]).build(
            s,
            super::conv2d::Conv2D {
                pad: self.pad,
                stride: self.stride,
                dilation: self.dilation,
            },
        );

        let gw = Tensor::builder()
            .set_inputs(&[&gy, &x, &s.stop_gradient(w)])
            .build(
                s,
                Conv2DTransposeFilterGrad {
                    pad: self.pad,
                    stride: self.stride,
                    dilation: self.dilation,
                },
            );

        ctx.set_input_grads(vec![Some(gx), Some(gw)]);
    }
}

impl<T: Float> crate::op::Op<T> for Conv2DTransposeFilterGrad {
    fn compute(&self, ctx: &mut crate::op::OpComputeContext<T>) {
        let gy = &ctx.input(0);
        let x = &ctx.input(1);
        let w = &ctx.input(2);
        let k_shape = w.shape();

        let x_shape = x.shape();
        let gy_shape = gy.shape();

        let batch_size = x_shape[0];
        let (kh, kw) = (k_shape[2], k_shape[3]);

        let size_per_batch_c = {
            get_yh!(self, gy_shape[2], kh) * get_yw!(self, gy_shape[3], kw) * kh * kw * gy_shape[1]
        };
        let size_per_batch_x = x_shape[1] * x_shape[2] * x_shape[3];

        // sgemm params
        let m = x_shape[1];
        let n = kh * kw * gy_shape[1];
        let k = get_yh!(self, gy_shape[2], kh) * get_yw!(self, gy_shape[3], kw);

        let x = unsafe { slice::from_raw_parts(x.as_ptr(), x.len()) };
        let gy = unsafe { slice::from_raw_parts(gy.as_ptr(), gy.len()) };

        let cols = im2col_batch(
            gy,
            batch_size as usize,
            gy_shape[1] as i32,
            gy_shape[2] as i32,
            gy_shape[3] as i32,
            kh as i32,
            kw as i32,
            self.pad as i32,
            self.pad as i32,
            self.stride as i32,
            self.stride as i32,
            self.dilation as i32,
            self.dilation as i32,
        );

        let gw = unsafe {
            let mut gw = uninitialized_vec(k_shape[0] * k_shape[1] * k_shape[2] * k_shape[3]);
            let gw_head = gw.as_mut_ptr();

            for i in 0..batch_size {
                let x_region_head = &x[i * size_per_batch_x] as *const T;
                let c_region_head = &cols[i * size_per_batch_c] as *const T;
                #[cfg(feature = "mkl")]
                {
                    if same_type::<T, f32>() {
                        crate::ops::dot_ops::cblas_sgemm_wrapper(
                            false,
                            true,
                            m,
                            n,
                            k,
                            1.,
                            x_region_head as *const f32,
                            c_region_head as *const f32,
                            if i == 1 { 1. } else { 0. },
                            gw_head as *mut f32,
                        )
                    } else if same_type::<T, f64>() {
                        crate::ops::dot_ops::cblas_dgemm_wrapper(
                            false,
                            true,
                            m,
                            n,
                            k,
                            1.,
                            x_region_head as *const f64,
                            c_region_head as *const f64,
                            if i == 1 { 1. } else { 0. },
                            gw_head as *mut f64,
                        )
                    }
                }
                #[cfg(not(feature = "mkl"))]
                {
                    slow_gemm!(
                        false,
                        true,
                        x_region_head as *const f32,
                        c_region_head as *const f32,
                        gw_head as *mut f32,
                        m,
                        n,
                        k,
                        1.,
                        if i == 1 { 1. } else { 0. }
                    );
                }
            }
            gw
        };

        ctx.push_output(Ok(crate::ArrRepr::Owned(
            NdArray::from_shape_vec(k_shape, gw).unwrap(),
        )));
    }

    fn grad(&self, ctx: &mut crate::op::OpGradientContext<T>) {
        let s = ctx.graph();
        let gy = ctx.input(0);
        let gw = ctx.output_grad();
        let x = ctx.input(1);

        let ggy = Tensor::builder().set_inputs(&[&x, &gw]).build(
            s,
            Conv2DTranspose {
                pad: self.pad,
                stride: self.stride,
                dilation: self.dilation,
            },
        );

        let ggx = Tensor::builder().set_inputs(&[&gy, &gw]).build(
            s,
            super::conv2d::Conv2D {
                pad: self.pad,
                stride: self.stride,
                dilation: self.dilation,
            },
        );

        ctx.set_input_grads(vec![Some(ggy), Some(ggx), None]);
    }
}

#[test]
fn test_tensor_size_after_convolution_t() {
    let op = Conv2DTranspose {
        pad: 0,
        stride: 1,
        dilation: 1,
    };
    let (yh, yw) = (2, 2);
    let (kh, kw) = (2, 2);
    let xh = get_xh!(&op, yh, kh);
    let xw = get_xw!(&op, yw, kw);
    assert_eq!(xh, 3);
    assert_eq!(xw, 3);
}

#[test]
fn test_deconv() {
    let (kh, kw) = (2, 2);
    let (xch, ych) = (3, 2);
    let (yh, yw) = (2, 2);
    let batch_size = 2;
    let ans = NdArray::<f32>::from_shape_vec(
        ndarray::IxDyn(&[2, 3, 3, 3]),
        vec![
            2.0, 4.0, 2.0, 4.0, 8.0, 4.0, 2.0, 4.0, 2.0, 2.0, 4.0, 2.0, 4.0, 8.0, 4.0, 2.0, 4.0,
            2.0, 2.0, 4.0, 2.0, 4.0, 8.0, 4.0, 2.0, 4.0, 2.0, 2.0, 4.0, 2.0, 4.0, 8.0, 4.0, 2.0,
            4.0, 2.0, 2.0, 4.0, 2.0, 4.0, 8.0, 4.0, 2.0, 4.0, 2.0, 2.0, 4.0, 2.0, 4.0, 8.0, 4.0,
            2.0, 4.0, 2.0,
        ],
    )
    .unwrap();

    crate::with::<f32, _>(|s| {
        let w = s.ones(&[ych, xch, kh, kw]);
        let g = s.ones(&[batch_size, ych, yh, yw]);
        let out = s.conv2d_transpose(g, w, 0, 1);
        let out_val = out.eval(&[]).unwrap();
        out_val.all_close(&ans, 1e-3);
    });
}
