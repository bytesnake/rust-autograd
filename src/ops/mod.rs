//! A collection of functions to manipulate `ag::Tensor` objects
use ndarray;

use crate::context::Context;
use crate::ndarray_ext::{ArrRng, NdArray};
use crate::tensor::{ArrayLike, Tensor};
use crate::Float;
use rand::Rng;

mod activation_ops;
mod array_ops;
mod basic_source_ops;
#[doc(hidden)]
pub mod binary_ops;
mod const_gen_ops;
mod conv_ops;
#[macro_use]
#[doc(hidden)]
pub mod dot_ops;
pub mod gradient_descent_ops;
mod gradient_ops;
#[doc(hidden)]
pub mod hook_ops;
mod math_ops;
mod random_ops;
mod reduction_ops;
mod xent_ops;

type T<'a, F> = &'a Tensor<'a, F>;
// ---------------------------------------
// -- Ops to manipulate `Tensor` object --
// ---------------------------------------

impl<'a, F: Float> Tensor<'a, F> {
    /// Looks up a symbolic element from this tensor.
    ///
    /// Index `i` can be negative.
    ///
    /// ```
    /// use ndarray;
    /// use autograd as ag;
    ///
    /// let ref a = ag::variable(ndarray::arr2(&[[2., 3.], [4., 5.]]));
    /// let ref b = a.get(2);
    ///
    /// assert_eq!(b.eval(&[]).unwrap()[ndarray::IxDyn(&[])], 4.);
    /// ```
    pub fn get(&'a self, i: isize, c: &mut Context<'a, F>) -> &'a Tensor<'a, F> {
        let op = array_ops::IndexOp { index: i };
        Tensor::builder().set_input(self).build(c, op)
    }
}

impl<'a, F: Float> crate::context::Context<'a, F> {
    /// Returns gradient tensors wrt input tensors.
    ///
    /// # Arguments
    /// * `ys` - Targets of differentiation that are arbitrary shapes.
    /// * `xs` - Tensors with which differentiate `ys`.
    ///
    /// # Returns
    /// Symbolic gradient tensors of `xs` in the same order as `xs`'s.
    ///
    /// # Example
    /// Partial derivatives of `z = 2x^2 + 3y + 1`.
    ///
    /// ```
    /// use ndarray;
    /// use autograd as ag;
    ///
    /// let ref x = ag::placeholder::<f64>(&[]);
    /// let ref y = ag::placeholder::<f64>(&[]);
    /// let ref z = 2.*x*x + 3.*y + 1.;
    ///
    /// // dz/dy
    /// let ref gy = ag::grad(&[z], &[y])[0];
    /// // dz/dx
    /// let ref gx = ag::grad(&[z], &[x])[0];
    ///
    /// // ddz/dx (differentiates `z` again)
    /// let ref ggx = ag::grad(&[gx], &[x])[0];
    ///
    /// // evaluation of symbolic gradients
    /// assert_eq!(3., gy.eval(&[]).unwrap()[ndarray::IxDyn(&[])]);
    /// assert_eq!(4., ggx.eval(&[]).unwrap()[ndarray::IxDyn(&[])]);
    ///
    /// // dz/dx requires to fill the placeholder `x`
    /// assert_eq!(8., gx.eval(&[ag::Feed(x, ndarray::arr0(2.).into_dyn().view())]).unwrap()[ndarray::IxDyn(&[])]);
    /// ```
    ///
    /// See also [grad_with_default](fn.grad_with_default.html).
    pub fn grad(&mut self, ys: &[T<'a, F>], xs: &[T<'a, F>]) -> Vec<&'a Tensor<'a, F>>
    {
        let ys = ys
            .into_iter()
            .map(|y| self.reduce_sum_to_scalar(y))
            .collect::<Vec<_>>();
        let gys = vec![self.scalar(F::one()); ys.len()];
        unsafe { self.grad_with_default(&ys, xs, &gys) }
    }

    /// Computes gradients with `ys`'s already known gradients.
    ///
    /// Almost same spec as [grad](fn.grad.html)'s except that you can pass `ys`s already known gradients.
    /// If `ys_grads` are tensors filled with 1s, this function should be replaced with [grad](fn.grad.html).
    ///
    /// NOTE: Please be careful to match `ys_grads[i].shape` and `ys[i].shape`,
    /// otherwise **undefined behavior** would happen.
    ///
    /// # Arguments
    /// * `ys` - Targets of differentiation.
    /// * `xs` - tensors with which differentiate `ys`.
    /// * `ys_grads` - Already known gradients of `ys`.
    ///
    /// # Returns
    /// Symbolic gradient tensors of `xs` in the same order as `xs`'s.
    pub unsafe fn grad_with_default(&mut self, ys: &[T<'a, F>], xs: &[T<'a, F>], ys_grads: &[T<'a, F>]) -> Vec<T<'a, F>>
    {
        crate::gradient::symbolic_gradients(xs, ys, ys_grads, self)
    }

    /// Computes jacobians for variables.
    ///
    /// # Arguments
    /// * `y` - Target of differentiation.
    /// * `xs` - Tensors with which differentiate `ys`.
    /// * `y_size` - (flattened) size of `y`
    ///
    /// # Returns
    /// Jacobians for each variable. Each one is a matrix of shape `(y_size, x size)`.
    ///
    /// ```
    /// use autograd as ag;
    ///
    /// let ref a = ag::variable(ag::ndarray_ext::standard_normal::<f32>(&[4, 2]));
    /// let ref b = ag::variable(ag::ndarray_ext::standard_normal::<f32>(&[2, 3]));
    /// let ref c = ag::matmul(a, b);
    /// let ref j = ag::jacobians(c, &[a, b], 4*3);
    ///
    /// assert_eq!(j[0].eval(&[]).unwrap().shape(), &[4*3, 4*2]);
    /// assert_eq!(j[1].eval(&[]).unwrap().shape(), &[4*3, 2*3]);
    /// ```
    pub fn jacobians(
        &mut self,
        y: &'a Tensor<'a, F>,
        xs: &[&'a Tensor<'a, F>],
        objective_len: usize,
    ) -> Vec<&'a Tensor<'a, F>> {
        let vec_vec = (0..objective_len as isize)
            .map(|i| {
                // For each scalar objective, computes gradients for all variables
                self.grad(&[&y.get(i, self)], xs)
            })
            .collect::<Vec<Vec<_>>>();

        // post process gradients
        (0..xs.len())
            .map(|i| {
                // jac is matrix
                let jac = (0..objective_len)
                    .map(|j| self.expand_dims(self.flatten(&vec_vec[j][i]), &[0]))
                    .collect::<Vec<_>>();
                // (y size, x size)
                self.concat(&jac, 0)
            })
            .collect::<Vec<_>>()
    }

    /// (Experimental) Computes hessian vector product
    pub fn _hessian_vector_product(
        &mut self,
        ys: &[&'a Tensor<'a, F>],
        xs: &[&'a Tensor<'a, F>],
        vectors: &[&'a Tensor<'a, F>],
    ) -> Vec<&'a Tensor<'a, F>> {
        let grads = self.grad(ys, xs);
        let products = grads
            .into_iter()
            .zip(vectors)
            .map(|(g, &v)| g * v)
            .collect::<Vec<_>>();
        self.grad(products.as_slice(), xs)
    }

    /// Stops gradient propagation.
    ///
    /// Guarantees that the gradient is not propagated to the tensors behind this
    /// during gradient computation.
    pub fn stop_gradient(&mut self, x: &'a Tensor<'a, F>) -> &'a Tensor<'a, F> {
        Tensor::builder()
            .set_input(x)
            .set_differentiable(false)
            .build(self, gradient_ops::StopGradient)
    }

    /// Creates a shared variable tensor from an ndarray.
    ///
    /// A shared variable can be mutated with gradient descent methods
    /// implemented in `autograd::gradient_descent_ops`.
    /// For the usages, see https://github.com/perrier1034/rust-autograd/tree/master/examples.
    ///
    /// ```
    /// use ndarray;
    /// use autograd as ag;
    ///
    /// let ref x: ag::Tensor<f64> = ag::variable(ndarray::arr1(&[2.]));
    /// let ref y: ag::Tensor<f64> = 3. * x;
    ///
    /// assert_eq!(6., y.eval(&[]).unwrap()[0]);
    /// ```
    #[inline]
    pub fn variable<D: ndarray::Dimension>(&mut self, arr: ndarray::Array<F, D>) -> &'a Tensor<'a, F> {
        let arr = arr.into_dyn();
        Tensor::builder()
            .set_shape(self.convert_to_tensor(crate::ndarray_ext::shape_of(&arr)))
            .set_variable_array(arr)
            .build(self, basic_source_ops::Variable)
    }

    /// Creates a placeholder tensor.
    ///
    /// Behaves like TensorFlow's placeholder object.
    /// `shape_[i]` must be positive value, or -1 which means dynamic dim.
    ///
    /// ```
    /// use ndarray;
    /// use autograd as ag;
    ///
    /// let x = ag::placeholder(&[2]);
    ///
    /// // Fills placeholder, then eval
    /// let arr = ndarray::arr1(&[1., 1.]).into_dyn();
    /// assert_eq!(x.eval(&[ag::Feed(&x, arr.clone().view())]), Some(arr));
    /// ```
    #[inline]
    pub fn placeholder(&mut self, shape_: &[isize]) -> &'a Tensor<'a, F> {
        let b = Tensor::builder().set_is_placeholder(true);
        let rank = shape_.len();
        let b = if rank == 0 || -1 != shape_[0] {
            b.set_shape(self.convert_to_tensor(
                NdArray::from_shape_vec(
                    ndarray::IxDyn(&[rank]),
                    shape_
                        .iter()
                        .map(|&x| F::from(x).unwrap())
                        .collect::<Vec<_>>(),
                )
                .unwrap(),
            ))
        } else {
            b
        };
        let b = b.set_known_shape(crate::tensor::KnownShape::new(shape_.to_vec()));
        b.build(self, basic_source_ops::Placeholder)
    }

    /// Creates a constant tensor.
    ///
    /// ```
    /// use ndarray;
    /// use autograd as ag;
    ///
    /// let arr = ndarray::arr1(&[0., 0., 0.]);
    /// let ref con = ag::constant(arr.clone());
    /// assert_eq!(con.eval(&[]), Some(arr.into_dyn()))
    /// ```
    #[inline]
    pub fn constant<D>(&mut self, arr: ndarray::Array<F, D>) -> &'a Tensor<'a, F>
    where
        D: ndarray::Dimension,
    {
        let arr = arr.into_dyn();
        Tensor::builder()
            .set_shape(self.convert_to_tensor(crate::ndarray_ext::shape_of(&arr)))
            .set_constant_array(arr)
            .build(self, basic_source_ops::Const)
    }

    /// Returns the (symbolic) shape of input tensor
    ///
    /// ```
    /// use autograd as ag;
    ///
    /// let ref x: ag::Tensor<f32> = ag::zeros(&[2, 3]);
    /// let ref s = ag::shape(x);
    ///
    /// assert_eq!(&[2., 3.], s.eval(&[]).unwrap().as_slice().unwrap());
    /// ```
    pub fn shape(&mut self, x: &'a Tensor<'a, F>) -> &'a Tensor<'a, F> {
        if let Some(ref inner) = x.shape {
            inner.clone()
        } else {
            Tensor::builder()
                .set_input(x)
                .set_differentiable(false)
                .build(self, array_ops::Shape)
        }
    }

    /// Returns the (symbolic) size of input tensor
    ///
    /// ```
    /// use ndarray;
    /// use autograd as ag;
    ///
    /// let ref a: ag::Tensor<f32> = ag::zeros(&[4, 3]);
    /// let ref b = ag::size(a);
    ///
    /// assert_eq!(12., b.eval(&[]).unwrap()[ndarray::IxDyn(&[])]);
    /// ```
    pub fn size(&mut self, x: &'a Tensor<'a, F>) -> &'a Tensor<'a, F> {
        Tensor::builder()
            .set_input(x)
            .set_differentiable(false)
            .build(self, array_ops::Size)
    }

    /// Returns the (symbolic) rank of input tensor
    ///
    /// ```
    /// use ndarray;
    /// use autograd as ag;
    ///
    /// let ref x: ag::Tensor<f32> = ag::zeros(&[2, 3, 4]);
    /// let ref r = ag::rank(x);
    ///
    /// assert_eq!(3., r.eval(&[]).unwrap()[ndarray::IxDyn(&[])]);
    /// ```
    pub fn rank(&mut self, x: &'a Tensor<'a, F>) -> &'a Tensor<'a, F> {
        Tensor::builder()
            .set_input(x)
            .set_differentiable(false)
            .build(self, array_ops::Rank)
    }

    /// Elementwise sine
    pub fn sin(&mut self, x: &'a Tensor<'a, F>) -> &'a Tensor<'a, F> {
        Tensor::builder()
            .set_input(x)
            .set_shape(self.shape(x))
            .build(self, math_ops::Sin)
    }

    /// Elementwise cosine
    pub fn cos(&mut self, x: &'a Tensor<'a, F>) -> &'a Tensor<'a, F> {
        Tensor::builder()
            .set_input(x)
            .set_shape(self.shape(x))
            .build(self, math_ops::Cos)
    }

    /// Elementwise tangent
    pub fn tan(&mut self, x: &'a Tensor<'a, F>) -> &'a Tensor<'a, F> {
        Tensor::builder()
            .set_input(x)
            .set_shape(self.shape(x))
            .build(self, math_ops::Tan)
    }

    /// Elementwise arcsin
    pub fn asin(&mut self, x: &'a Tensor<'a, F>) -> &'a Tensor<'a, F> {
        Tensor::builder()
            .set_input(x)
            .set_shape(self.shape(x))
            .build(self, math_ops::Asin)
    }

    /// Elementwise arccos
    pub fn acos(&mut self, x: &'a Tensor<'a, F>) -> &'a Tensor<'a, F> {
        Tensor::builder()
            .set_input(x)
            .set_shape(self.shape(x))
            .build(self, math_ops::Acos)
    }

    /// Elementwise arctan
    pub fn atan(&mut self, x: &'a Tensor<'a, F>) -> &'a Tensor<'a, F> {
        Tensor::builder()
            .set_input(x)
            .set_shape(self.shape(x))
            .build(self, math_ops::Atan)
    }

    /// Elementwise hyperbolic sine
    pub fn sinh(&mut self, x: &'a Tensor<'a, F>) -> &'a Tensor<'a, F> {
        Tensor::builder()
            .set_input(x)
            .set_shape(self.shape(x))
            .build(self, math_ops::Sinh)
    }

    /// Elementwise hyperbolic cosine
    pub fn cosh(&mut self, x: &'a Tensor<'a, F>) -> &'a Tensor<'a, F> {
        Tensor::builder()
            .set_input(x)
            .set_shape(self.shape(x))
            .build(self, math_ops::Cosh)
    }

    /// Elementwise hyperbolic tangent
    pub fn tanh(&mut self, x: &'a Tensor<'a, F>) -> &'a Tensor<'a, F> {
        Tensor::builder()
            .set_input(x)
            .set_shape(self.shape(x))
            .build(self, math_ops::Tanh)
    }

    /// Elementwise hyperbolic arcsin
    pub fn asinh(&mut self, x: &'a Tensor<'a, F>) -> &'a Tensor<'a, F> {
        Tensor::builder()
            .set_input(x)
            .set_shape(self.shape(x))
            .build(self, math_ops::Asinh)
    }

    /// Elementwise hyperbolic arccos
    pub fn acosh(&mut self, x: &'a Tensor<'a, F>) -> &'a Tensor<'a, F> {
        Tensor::builder()
            .set_input(x)
            .set_shape(self.shape(x))
            .build(self, math_ops::Acosh)
    }

    /// Elementwise hyperbolic arctan
    pub fn atanh(&mut self, x: &'a Tensor<'a, F>) -> &'a Tensor<'a, F> {
        Tensor::builder()
            .set_input(x)
            .set_shape(self.shape(x))
            .build(self, math_ops::Atanh)
    }

    #[doc(hidden)]
    /// Gets n th tensor in `x`.
    ///
    /// `x` must be a result of a multi-outputs op;
    /// otherwise index-out-of-bounds error may happen.
    pub fn nth_tensor(&mut self, x: &'a Tensor<'a, F>, n: usize) -> &'a Tensor<'a, F>
    {
        Tensor::builder()
            .set_input(x)
            .set_input_indices(vec![n])
            .build(self, activation_ops::Identity)
    }

    /// Identity function without copy.
    pub fn identity(&mut self, x: &'a Tensor<'a, F>) -> &'a Tensor<'a, F> {
        Tensor::builder()
            .set_input(x)
            .set_shape(self.shape(x))
            .build(self, activation_ops::Identity)
    }

    #[inline]
    fn infer_bin_op_shape(
        &mut self,
        shape_a: &'a Tensor<'a, F>,
        shape_b: &'a Tensor<'a, F>
    ) -> &'a Tensor<'a, F> {
        Tensor::builder()
            .set_inputs(&[shape_a, shape_b])
            .build(self, array_ops::InferBinOpShape)
    }

    #[inline]
    fn bin_op_helper<O: crate::op::Op<'a, F> + Send + Sync + 'static>(
        &mut self,
        a: &'a Tensor<'a, F>,
        b: &'a Tensor<'a, F>, 
        op: O,
    ) -> &'a Tensor<'a, F> {
        let a_shape = self.shape(a);
        let b_shape = self.shape(b);
        Tensor::builder()
            .set_shape(self.infer_bin_op_shape(&a_shape, &b_shape))
            .set_inputs(&[a, b])
            .build(self, op)
    }

    /// Addition.
    ///
    /// `+` operator can be used instead.
    ///
    /// ```
    /// use ndarray;
    /// use autograd as ag;
    ///
    /// let ref a = ag::ones(&[2]);
    /// let ref b = ag::ones(&[2]);
    /// let ref z: ag::Tensor<f32> = a + b;
    /// assert_eq!(z.eval(&[]), Some(ndarray::arr1(&[2., 2.]).into_dyn()));
    /// ```
    pub fn add(&mut self, a: &'a Tensor<'a, F>, b: &'a Tensor<'a, F>) -> &'a Tensor<'a, F> {
        self.bin_op_helper(a, b, binary_ops::AddOp)
    }

    /// Subtraction.
    ///
    /// `-` operator can be used instead.
    ///
    /// ```
    /// use ndarray;
    /// use autograd as ag;
    ///
    /// let ref a = ag::ones(&[2]);
    /// let ref b = ag::ones(&[2]);
    ///
    /// let ref z: ag::Tensor<f32> = a - b;
    /// assert_eq!(z.eval(&[]), Some(ndarray::arr1(&[0., 0.]).into_dyn()));
    /// ```
    pub fn sub(&mut self, a: &'a Tensor<'a, F>, b: &'a Tensor<'a, F>) -> &'a Tensor<'a, F> {
        self.bin_op_helper(a, b, binary_ops::SubOp)
    }

    /// Multiplication.
    ///
    /// `*` operator can be used instead.
    ///
    /// ```
    /// use ndarray;
    /// use autograd as ag;
    ///
    ///
    /// let ref a = ag::ones(&[2]);
    /// let ref b = ag::ones(&[2]);
    /// let ref z: ag::Tensor<f32> = a * b;
    /// assert_eq!(z.eval(&[]), Some(ndarray::arr1(&[1., 1.]).into_dyn()));
    /// ```
    pub fn mul(&mut self, a: &'a Tensor<'a, F>, b: &'a Tensor<'a, F>) -> &'a Tensor<'a, F> {
        self.bin_op_helper(a, b, binary_ops::MulOp)
    }

    /// Division.
    ///
    /// `/` operator can be used instead.
    ///
    /// ```
    /// use ndarray;
    /// use autograd as ag;
    ///
    /// let ref a = ag::ones(&[2]);
    /// let ref b = ag::ones(&[2]);
    /// let ref z: ag::Tensor<f32> = a / b;
    /// assert_eq!(z.eval(&[]), Some(ndarray::arr1(&[1., 1.]).into_dyn()));
    /// ```
    pub fn div(&mut self, a: &'a Tensor<'a, F>, b: &'a Tensor<'a, F>) -> &'a Tensor<'a, F> {
        self.bin_op_helper(a, b, binary_ops::DivOp)
    }

    /// Elementwise sqrt
    pub fn sqrt(&mut self, x: &'a Tensor<'a, F>) -> &'a Tensor<'a, F> {
        Tensor::builder()
            .set_input(x)
            .set_shape(self.shape(x))
            .build(self, math_ops::Sqrt)
    }

    /// Elementwise pow
    pub fn pow(&mut self, x: &'a Tensor<'a, F>, a: F) -> &'a Tensor<'a, F> {
        Tensor::builder()
            .set_input(x)
            .set_shape(self.shape(x))
            .build(self, math_ops::Pow { a })
    }

    /// Elementwise log
    pub fn log(&mut self, x: &'a Tensor<'a, F>, a: F) -> &'a Tensor<'a, F> {
        Tensor::builder()
            .set_input(x)
            .set_shape(self.shape(x))
            .build(self, math_ops::Log { a })
    }

    /// Elementwise exponential
    pub fn exp(&mut self, x: &'a Tensor<'a, F>) -> &'a Tensor<'a, F> {
        Tensor::builder()
            .set_input(x)
            .set_shape(self.shape(x))
            .build(self, math_ops::Exp)
    }

    /// Returns the max of x and y (i.e. x > y ? x : y) element-wise.
    ///
    /// ```
    /// use ndarray;
    /// use autograd as ag;
    ///
    /// let ref a = ag::constant(ndarray::arr1(&[1., 2., 3.]));
    /// let ref b = ag::constant(ndarray::arr1(&[3., 2., 1.]));
    /// let ref c = ag::maximum(a, b);
    /// assert_eq!(c.eval(&[]), Some(ndarray::arr1(&[3., 2., 3.]).into_dyn()));
    /// ```
    pub fn maximum(&mut self, a: &'a Tensor<'a, F>, b: &'a Tensor<'a, F>) -> &'a Tensor<'a, F> {
        Tensor::builder()
            .set_inputs(&[a, b])
            .build(self, math_ops::Maximum)
    }

    /// Returns the min of x and y (i.e. x > y ? y : x) element-wise.
    ///
    /// ```
    /// use ndarray;
    /// use autograd as ag;
    ///
    /// let ref a = ag::constant(ndarray::arr1(&[1., 2., 3.]));
    /// let ref b = ag::constant(ndarray::arr1(&[3., 2., 1.]));
    /// let ref c = ag::minimum(a, b);
    /// assert_eq!(c.eval(&[]), Some(ndarray::arr1(&[1., 2., 1.]).into_dyn()));
    /// ```
    pub fn minimum(&mut self, a: &'a Tensor<'a, F>, b: &'a Tensor<'a, F>) -> &'a Tensor<'a, F> {
        Tensor::builder()
            .set_inputs(&[a, b])
            .build(self, math_ops::Minimum)
    }

    /// Adds all input tensors, element-wise.
    ///
    /// All the input tensors must have same shapes.
    ///
    /// ```
    /// use ndarray;
    /// use autograd as ag;
    ///
    /// let ref a = ag::ones(&[2, 2]);
    /// let ref b = ag::ones(&[2, 2]);
    /// let ref c = ag::ones(&[2, 2]);
    /// let ref d = ag::add_n(&[a, b, c]);
    ///
    /// assert_eq!(d.eval(&[]).unwrap().shape(), &[2, 2]);
    /// assert_eq!(d.eval(&[]), Some(ndarray::arr2(&[[3., 3.], [3., 3.]]).into_dyn()));
    /// ```
    pub fn add_n(&mut self, xs: &[&'a Tensor<'a, F>]) -> &'a Tensor<'a, F> {
        let len = xs.len();
        assert_ne!(len, 0);
        if len == 1 {
            xs[0].clone()
        } else {
            Tensor::builder()
                .set_inputs(xs)
                .set_shape(self.shape(xs[0]))
                .build(self, array_ops::AddN)
        }
    }

    /// Compares two tensors and returns a binary tensor.
    ///
    /// if `a[i] == b[i]` then `return-value[i]` will be 1 else 0
    ///
    /// # Panics
    /// When broadcast is impossible
    ///
    /// ```
    /// use ndarray;
    /// use autograd as ag;
    ///
    /// let ref a = ag::constant(ndarray::arr1(&[1., 2., 3.]));
    /// let ref b = ag::constant(ndarray::arr1(&[3., 2., 1.]));
    /// let ref c = ag::equal(a, b);
    ///
    /// assert_eq!(c.eval(&[]), Some(ndarray::arr1(&[0., 1., 0.]).into_dyn()));
    /// ```
    pub fn equal(&mut self, a: &'a Tensor<'a, F>, b: &'a Tensor<'a, F>) -> &'a Tensor<'a, F> {
        Tensor::builder()
            .set_inputs(&[a, b])
            .build(self, math_ops::Equal)
    }

    /// Compares two tensors and returns a binary tensor.
    ///
    /// if `a[i] != b[i]` then `return-value[i]` will be 1 else 0
    ///
    /// # Panics
    /// When broadcast is impossible
    ///
    /// ```
    /// use ndarray;
    /// use autograd as ag;
    ///
    /// let ref a = ag::constant(ndarray::arr1(&[1., 2., 3.]));
    /// let ref b = ag::constant(ndarray::arr1(&[3., 2., 1.]));
    /// let ref c = ag::not_equal(a, b);
    ///
    /// assert_eq!(c.eval(&[]), Some(ndarray::arr1(&[1., 0., 1.]).into_dyn()));
    /// ```
    pub fn not_equal(&mut self, a: &'a Tensor<'a, F>, b: &'a Tensor<'a, F>) -> &'a Tensor<'a, F> {
        Tensor::builder()
            .set_inputs(&[a, b])
            .build(self, math_ops::NotEqual)
    }

    /// Takes argmax along specified axis.
    ///
    /// `axis` can be negative.
    ///
    /// ```
    /// use ndarray;
    /// use autograd as ag;
    ///
    /// let ref x = ag::constant(ndarray::arr2(&[[3., 4.], [6., 5.]]));
    /// let ref y = ag::argmax(x, 1, false);
    ///
    /// assert_eq!(y.eval(&[]), Some(ndarray::arr1(&[1., 0.]).into_dyn()));
    /// ```
    pub fn argmax(&mut self, x: &'a Tensor<'a, F>, axis: isize, keep_dim: bool) -> &'a Tensor<'a, F> {
        let op = reduction_ops::ArgMax { axis, keep_dim };
        Tensor::builder().set_input(x).build(self, op)
    }

    /// Expands specified dims.
    ///
    /// Each axis can be negative.
    ///
    /// ```
    /// use autograd as ag;
    ///
    /// let ref a: ag::Tensor<f32> = ag::zeros(&[3]);
    /// let ref b = ag::expand_dims(a, &[0, 2]);
    ///
    /// assert_eq!(b.eval(&[]).unwrap().shape(), &[1, 3, 1]);
    /// ```
    pub fn expand_dims<AL: ArrayLike<'a, F>>(&mut self, x: &'a Tensor<'a, F>, axes: &AL) -> &'a Tensor<'a, F> {
        Tensor::builder()
            .set_inputs(&[x, axes.as_tensor(self)])
            .build(self, array_ops::ExpandDims)
    }

    /// Squeezes specified dims.
    ///
    /// Each axis can be negative.
    ///
    /// ```
    /// use autograd as ag;
    ///
    /// let ref a: ag::Tensor<f32> = ag::zeros(&[1, 3, 1]);
    /// let ref b = ag::squeeze(a, &[0, 2]);
    ///
    /// assert_eq!(b.eval(&[]).unwrap().shape(), &[3]);
    /// ```
    pub fn squeeze<AL: ArrayLike<'a, F>>(&mut self, x: &'a Tensor<'a, F>, axes: &AL) -> &'a Tensor<'a, F> {
        let op = array_ops::Squeeze;
        Tensor::builder()
            .set_inputs(&[x, axes.as_tensor(self)])
            .build(self, op)
    }

    /// Tiles input tensor along specified axis.
    ///
    /// Tiles input tensor `num` times along `axis`.
    /// `axis` can be negative.
    ///
    /// ```
    /// use ndarray;
    /// use autograd as ag;
    ///
    /// let ref x = ag::constant(ndarray::arr2(&[[2., 2.], [3., 3.]]));
    /// let ref y = ag::tile(x, 0, 2);
    ///
    /// assert_eq!(
    ///     y.eval(&[]),
    ///     Some(ndarray::arr2(&[[2., 2.], [3., 3.], [2., 2.], [3., 3.]]).into_dyn())
    /// );
    /// ```
    pub fn tile(&mut self, x: &'a Tensor<'a, F>, axis: isize, num: usize) -> &'a Tensor<'a, F> {
        let op = array_ops::Tile { axis, num };
        Tensor::builder().set_input(x).build(self, op)
    }

    /// Limits all elements of `x` so as to be within `[min, max]`
    ///
    /// ```
    /// use ndarray;
    /// use autograd as ag;
    ///
    /// let ref x = ag::constant(ndarray::arr1(&[2., 4., 6.]));
    /// let ref y = ag::clip(x, 3., 5.);
    ///
    /// assert_eq!(y.eval(&[]), Some(ndarray::arr1(&[3., 4., 5.]).into_dyn()));
    /// ```
    pub fn clip(&mut self, x: &'a Tensor<'a, F>, min: F, max: F) -> &'a Tensor<'a, F> {
        let op = array_ops::Clip { min, max };
        Tensor::builder().set_input(x).build(self, op)
    }

    /// Takes max along specified axes.
    ///
    /// Elements of `axes` can be negative.
    ///
    /// ```
    /// use ndarray;
    /// use autograd as ag;
    ///
    /// let ref x = ag::constant(ndarray::arr2(&[[2., 4.], [3., 1.]]));
    /// let ref y = ag::reduce_max(&x, &[0], false);
    ///
    /// assert_eq!(y.eval(&[]), Some(ndarray::arr1(&[3., 4.]).into_dyn()));
    /// ```
    pub fn reduce_max<AL: ArrayLike<'a, F>>(
        &mut self,
        x: &'a Tensor<'a, F>,
        axes: &AL,
        keep_dims: bool,
    ) -> &'a Tensor<'a, F> {
        let op = reduction_ops::ReduceMax {
            keep_dims,
            sparse_axes: false,
        };
        Tensor::builder()
            .set_inputs(&[x, axes.as_tensor(self)])
            .build(self, op)
    }

    /// Takes min along specified axes.
    ///
    /// Elements of `axes` can be negative.
    ///
    /// ```
    /// use ndarray;
    /// use autograd as ag;
    ///
    /// let ref x = ag::constant(ndarray::arr2(&[[2., 4.], [3., 1.]]));
    /// let ref y = ag::reduce_min(&x, &[0], false);
    ///
    /// assert_eq!(y.eval(&[]), Some(ndarray::arr1(&[2., 1.]).into_dyn()));
    /// ```
    pub fn reduce_min<AL: ArrayLike<'a, F>>(
        &mut self,
        x: &'a Tensor<'a, F>,
        axes: &AL,
        keep_dims: bool,
    ) -> &'a Tensor<'a, F> {
        let op = reduction_ops::ReduceMin {
            keep_dims,
            sparse_axes: false,
        };
        Tensor::builder()
            .set_inputs(&[x, axes.as_tensor(self)])
            .build(self, op)
    }

    /// Sum up all the elements to a scalar value (0-D Tensor).
    ///
    /// ```
    /// use ndarray;
    /// use autograd as ag;
    ///
    /// let ref x = ag::constant(ndarray::arr2(&[[2., 4.], [3., 1.]]));
    /// let ref y = ag::reduce_sum_to_scalar(&x);
    ///
    /// assert_eq!(y.eval(&[]), Some(ndarray::arr0(10.).into_dyn()));
    /// ```
    pub fn reduce_sum_to_scalar(&mut self, x: &'a Tensor<'a, F>) -> &'a Tensor<'a, F> {
        Tensor::builder()
            .set_input(x)
            .build(self, reduction_ops::ReduceSumToScalar)
    }

    /// Takes sum along specified axes.
    ///
    /// Elements of `axes` can be negative.
    ///
    /// ```
    /// use ndarray;
    /// use autograd as ag;
    ///
    /// let ref x = ag::constant(ndarray::arr2(&[[2., 4.], [3., 1.]]));
    /// let ref y = ag::reduce_sum(&x, &[1], false);
    ///
    /// assert_eq!(y.eval(&[]), Some(ndarray::arr1(&[6., 4.]).into_dyn()));
    /// ```
    pub fn reduce_sum<AL: ArrayLike<'a, F>>(
        &mut self,
        x: &'a Tensor<'a, F>,
        axes: &AL,
        keep_dims: bool,
    ) -> &'a Tensor<'a, F> {
        let op = reduction_ops::ReduceSum {
            keep_dims,
            sparse_axes: false,
        };
        Tensor::builder()
            .set_inputs(&[x, axes.as_tensor(self)])
            .build(self, op)
    }

    /// Takes mean along specified axes.
    ///
    /// Elements of `axes` can be negative.
    ///
    /// ```
    /// use ndarray;
    /// use autograd as ag;
    ///
    /// let ref x = ag::constant(ndarray::arr2(&[[2., 4.], [3., 1.]]));
    /// let ref y = ag::reduce_mean(x, &[1], false);
    ///
    /// assert_eq!(y.eval(&[]), Some(ndarray::arr1(&[3., 2.]).into_dyn()));
    /// ```
    pub fn reduce_mean<AL: ArrayLike<'a, F>>(
        &mut self,
        x: &'a Tensor<'a, F>,
        axes: &AL,
        keep_dims: bool,
    ) -> &'a Tensor<'a, F> {
        let op = reduction_ops::ReduceMean {
            keep_dims,
            sparse_axes: false,
        };
        Tensor::builder()
            .set_inputs(&[x, axes.as_tensor(self)])
            .build(self, op)
    }

    /// Takes product along specified axes.
    ///
    /// Elements of `axes` can be negative.
    ///
    /// ```
    /// use ndarray;
    /// use autograd as ag;
    ///
    /// let ref x = ag::constant(ndarray::arr2(&[[2., 4.], [3., 1.]]));
    /// let ref y = ag::reduce_prod(&x, &[1], false);
    ///
    /// assert_eq!(y.eval(&[]), Some(ndarray::arr1(&[8., 3.]).into_dyn()));
    /// ```
    pub fn reduce_prod<AL: ArrayLike<'a, F>>(
        &mut self,
        x: &'a Tensor<'a, F>,
        axes: &AL,
        keep_dims: bool,
    ) -> &'a Tensor<'a, F> {
        let op = reduction_ops::ReduceProd {
            keep_dims,
            sparse_axes: false,
        };
        Tensor::builder()
            .set_inputs(&[x, axes.as_tensor(self)])
            .build(self, op)
    }

    /// Reshapes input tensor.
    ///
    /// Only one element in `shape` can be `-1`.
    ///
    /// ```
    /// use ndarray;
    /// use autograd as ag;
    ///
    /// let ref x: ag::Tensor<f32> = ag::zeros(&[3, 2, 2]);
    /// let ref y = ag::reshape(&x, &[3, -1]);
    ///
    /// assert_eq!(y.eval(&[]), Some(ag::ndarray_ext::zeros::<f32>(&[3, 4])));
    /// ```
    pub fn reshape<AL: ArrayLike<'a, F>>(&mut self, x: &'a Tensor<'a, F>, shape: &AL) -> &'a Tensor<'a, F> {
        Tensor::builder()
            .set_inputs(&[x, shape.as_tensor(self)])
            .build(self, array_ops::Reshape)
    }

    /// Flattens input tensor into 1-ranked (vector).
    ///
    /// ```
    /// use autograd as ag;
    ///
    /// let ref x: ag::Tensor<f32> = ag::zeros(&[3, 2, 2]);
    /// let ref z = ag::flatten(x);
    /// assert_eq!(z.eval(&[]).unwrap().shape(), &[12]);
    /// ```
    pub fn flatten(&mut self, x: &'a Tensor<'a, F>) -> &'a Tensor<'a, F> {
        Tensor::builder()
            .set_inputs(&[x, self.scalar(F::one().neg())])
            .set_shape(self.shape(x))
            .build(self, array_ops::Reshape)
    }

    /// Returns -1 if x < 0, 0 if x==0, 1 if x > 0, element-wise.
    ///
    /// ```
    /// use ndarray;
    /// use autograd as ag;
    ///
    /// let ref a = ag::constant(ndarray::arr1(&[-5., 4.5, 0.]));
    /// let ref b = ag::sign(a);
    /// assert_eq!(
    ///     b.eval(&[]).unwrap().as_slice().unwrap(),
    ///     &[-1., 1., 0.]
    /// );
    /// ```
    pub fn sign(&mut self, a: &'a Tensor<'a, F>) -> &'a Tensor<'a, F> {
        Tensor::builder()
            .set_shape(self.shape(a))
            .set_input(a)
            .build(self, math_ops::Sign)
    }

    /// Returns the largest integer less than or equal to a number, element-wise.
    ///
    /// ```
    /// use ndarray;
    /// use autograd as ag;
    ///
    /// let ref a = ag::constant(ndarray::arr1(&[-0.2, 0., 0.2]));
    /// let ref b = ag::abs(a);
    /// assert_eq!(
    ///     b.eval(&[]),
    ///     Some(ndarray::arr1(&[0.2, 0., 0.2]).into_dyn())
    /// );
    /// ```
    pub fn abs(&mut self, a: &'a Tensor<'a, F>) -> &'a Tensor<'a, F> {
        Tensor::builder()
            .set_shape(self.shape(a))
            .set_input(a)
            .build(self, math_ops::Abs)
    }

    /// Returns the largest integer less than or equal to a number, element-wise.
    ///
    /// ```
    /// use ndarray;
    /// use autograd as ag;
    ///
    /// let ref a = ag::constant(ndarray::arr1(&[-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0]));
    /// let ref b = ag::floor(a);
    /// assert_eq!(
    ///     b.eval(&[]),
    ///     Some(ndarray::arr1(&[-2., -2., -1.,  0.,  1.,  1.,  2.]).into_dyn())
    /// );
    /// ```
    pub fn floor(&mut self, a: &'a Tensor<'a, F>) -> &'a Tensor<'a, F> {
        Tensor::builder()
            .set_shape(self.shape(az))
            .set_input(a)
            .build(self, math_ops::Floor)
    }

    /// Performs the `-` operation.
    ///
    /// ```
    /// use ndarray;
    /// use autograd as ag;
    ///
    /// let ref a = ag::constant(ndarray::arr1(&[2., 3.]));
    /// let ref b = ag::neg(a);
    /// assert_eq!(
    ///     b.eval(&[]),
    ///     Some(ndarray::arr1(&[-2., -3.]).into_dyn())
    /// );
    /// ```
    pub fn neg(&mut self, a: &'a Tensor<'a, F>) -> &'a Tensor<'a, F> {
        Tensor::builder()
            .set_shape(self.shape(a))
            .set_input(a)
            .build(self, math_ops::NegOp)
    }

    /// Takes square of the input.
    ///
    /// ```
    /// use ndarray;
    /// use autograd as ag;
    ///
    /// let ref a = ag::constant(ndarray::arr1(&[2., 3.]));
    /// let ref b = ag::square(a);
    /// assert_eq!(
    ///     b.eval(&[]),
    ///     Some(ndarray::arr1(&[4., 9.]).into_dyn())
    /// );
    /// ```
    pub fn square(&mut self, a: &'a Tensor<'a, F>) -> &'a Tensor<'a, F> {
        Tensor::builder()
            .set_shape(self.shape(a))
            .set_input(a)
            .build(self, math_ops::Square)
    }

    /// Returns the 1/x, element-wise.
    ///
    /// ```
    /// use ndarray;
    /// use autograd as ag;
    ///
    /// let ref a = ag::constant(ndarray::arr1(&[2.]));
    /// let ref b = ag::reciprocal(a);
    /// assert_eq!(
    ///     b.eval(&[]),
    ///     Some(ndarray::arr1(&[0.5]).into_dyn())
    /// );
    /// ```
    pub fn reciprocal(&mut self, x: &'a Tensor<'a, F>) -> &'a Tensor<'a, F> {
        Tensor::builder()
            .set_shape(self.shape(x))
            .set_input(x)
            .build(self, math_ops::Reciprocal)
    }

    /// Returns the smallest integer greater than or equal to a number, element-wise.
    ///
    /// ```
    /// use ndarray;
    /// use autograd as ag;
    ///
    /// let ref a = ag::constant(ndarray::arr1(&[-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0]));
    /// let ref b = ag::ceil(a);
    /// assert_eq!(
    ///     b.eval(&[]),
    ///     Some(ndarray::arr1(&[-1., -1., -0.,  1.,  2.,  2.,  2.]).into_dyn())
    /// );
    /// ```
    pub fn ceil(&mut self, a: &'a Tensor<'a, F>) -> &'a Tensor<'a, F> {
        Tensor::builder()
            .set_shape(self.shape(a))
            .set_input(a)
            .build(self, math_ops::Ceil)
    }

    /// Returns a binary tensor.
    ///
    /// # Panics
    /// When broadcast is impossible
    pub fn greater(&mut self, a: &'a Tensor<'a, F>, b: &'a Tensor<'a, F>) -> &'a Tensor<'a, F> {
        Tensor::builder()
            .set_inputs(&[a, b])
            .build(self, math_ops::Greater)
    }

    /// Returns a binary tensor.
    ///
    /// # Panics
    /// When broadcast is impossible
    pub fn greater_equal(&mut self, a: &'a Tensor<'a, F>, b: &'a Tensor<'a, F>) -> &'a Tensor<'a, F> {
        Tensor::builder()
            .set_inputs(&[a, b])
            .build(self, math_ops::GreaterEqual)
    }

    /// Returns a binary tensor.
    ///
    /// # Panics
    /// When broadcast is impossible
    pub fn lesser(&mut self, a: &'a Tensor<'a, F>, b: &'a Tensor<'a, F>) -> &'a Tensor<'a, F> {
        Tensor::builder()
            .set_inputs(&[a, b])
            .build(self, math_ops::Lesser)
    }

    /// Returns a binary tensor.
    ///
    /// # Panics
    /// When broadcast is impossible
    pub fn lesser_equal(&mut self, a: &'a Tensor<'a, F>, b: &'a Tensor<'a, F>) -> &'a Tensor<'a, F> {
        Tensor::builder()
            .set_inputs(&[a, b])
            .build(self, math_ops::LesserEqual)
    }

    /// Elementwise logistic sigmoid function.
    pub fn sigmoid(&mut self, x: &'a Tensor<'a, F>) -> &'a Tensor<'a, F> {
        Tensor::builder()
            .set_shape(self.shape(x))
            .set_input(x)
            .build(self, activation_ops::Sigmoid)
    }

    /// Elementwise exponential linear unit.
    ///
    /// See https://arxiv.org/abs/1511.07289
    pub fn elu(&mut self, x: &'a Tensor<'a, F>, alpha: F) -> &'a Tensor<'a, F> {
        Tensor::builder()
            .set_shape(self.shape(x))
            .set_input(x)
            .build(self, activation_ops::ELU { alpha })
    }

    /// Elementwise rectified linear unit.
    pub fn relu(&mut self, x: &'a Tensor<'a, F>) -> &'a Tensor<'a, F> {
        Tensor::builder()
            .set_shape(self.shape(x))
            .set_input(x)
            .build(self, activation_ops::ReLU)
    }

    /// Elementwise leaky relu.
    ///
    /// In common, `alpha` is around 0.1 ~ 0.2.
    ///
    /// See http://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf
    pub fn leaky_relu(&mut self, x: &'a Tensor<'a, F>, alpha: F) -> &'a Tensor<'a, F> {
        self.maximum(&x, self.scalar(alpha) * x)
    }

    /// Elementwise softplus.
    pub fn softplus(&mut self, x: &'a Tensor<'a, F>) -> &'a Tensor<'a, F> {
        Tensor::builder()
            .set_shape(self.shape(x))
            .set_input(x)
            .build(self, activation_ops::Softplus)
    }

    /// Computes `log(sum(exp(x)))` along specified axis.
    ///
    /// `axis` can be negative.
    pub fn reduce_logsumexp(
        &mut self,
        x: &'a Tensor<'a, F>,
        axis: isize,
        keep_dim: bool,
    ) -> &'a Tensor<'a, F> {
        let op = math_ops::LogSumExp {
            axis,
            keep_dims: keep_dim,
        };
        Tensor::builder().set_input(x).build(self, op)
    }

    /// Log softmax function.
    ///
    /// Computes `softmax(x)` along specified axis and
    /// takes logarithm of it.
    /// `axis` can be negative.
    pub fn log_softmax(&mut self, x: &'a Tensor<'a, F>, axis: isize) -> &'a Tensor<'a, F> {
        Tensor::builder()
            .set_shape(self.shape(x))
            .set_input(x)
            .build(self, xent_ops::LogSoftmax { axis })
    }

    /// Computes softmax along specified axis
    ///
    /// `axis` can be negative.
    pub fn softmax(&mut self, x: &'a Tensor<'a, F>, axis: isize) -> &'a Tensor<'a, F> {
        let op = activation_ops::Softmax { axis };
        Tensor::builder().set_input(x).build(self, op)
    }

    /// Computes `binary_cross_entropy(sigmoid(y), t)`.
    ///
    /// This function is better than that combination in that it can prevent
    /// underflow of `log(sigmoid)`.
    ///
    /// # Arguments
    /// * `y` - Tensor with arbitrary shape
    /// * `t` - Ground-truth Tensor with same shape as `y`'s
    ///
    /// # Panics
    /// When y.shape != t.shape.
    ///
    /// # Returns
    /// Loss tensor with same shape as inputs's shapes
    pub fn sigmoid_cross_entropy(
        &mut self,
        y: &'a Tensor<'a, F>,
        t: &'a Tensor<'a, F>
    ) -> &'a Tensor<'a, F> {
        let op = xent_ops::SigmoidCrossEntropy;
        Tensor::builder()
            .set_shape(self.shape(y))
            .set_inputs(&[y, t])
            .build(self, op)
    }

    /// Computes `categorical_cross_entropy(softmax(y), t)`.
    ///
    /// This function is better than that combination in that it can prevent
    /// underflow of `log(softmax)`.
    ///
    /// # Arguments
    /// * `y` - Tensor with shape (batch_size, num_classes)
    /// * `t` - Tensor with shape (batch_size, num_classes)
    ///
    /// # Returns
    /// Loss tensor with shape (batch_size, 1)
    pub fn softmax_cross_entropy(
        &mut self,
        y: &'a Tensor<'a, F>,
        t: &'a Tensor<'a, F>
    ) -> &'a Tensor<'a, F> {
        let op = xent_ops::SoftmaxCrossEntropy;
        Tensor::builder()
            .set_inputs(&[y, t])
            .build(self, op)
    }

    /// A variant of `softmax_cross_entropy`.
    ///
    /// The behavior of this function is same as `softmax_cross_entropy`
    /// except that `t` is **not** batch of one-hot distributions but batch of ground truth label ids.
    ///
    /// # Arguments
    /// * `y` - Tensor with shape (batch_size, num_classes)
    /// * `t` - Tensor with shape (batch_size,) or (batch_size, 1)
    ///
    /// # Returns
    /// Loss tensor with shape (batch_size, 1)
    pub fn sparse_softmax_cross_entropy(
        &mut self,
        y: &'a Tensor<'a, F>,
        t: &'a Tensor<'a, F>
    ) -> &'a Tensor<'a, F> {
        let op = xent_ops::SparseSoftmaxCrossEntropy;
        Tensor::builder()
            .set_inputs(&[y, t])
            .build(self, op)
    }

    /// Matrix multiplication.
    ///
    /// Both `a` and `b` must be 2-ranked tensors.
    ///
    /// ```
    /// use autograd as ag;
    ///
    /// let ref a: ag::Tensor<f32> = ag::zeros(&[4, 2]);
    /// let ref b: ag::Tensor<f32> = ag::zeros(&[2, 3]);
    /// let ref c = ag::matmul(a, b);
    ///
    /// assert_eq!(c.eval(&[]).unwrap().shape(), &[4, 3]);
    /// ```
    ///
    /// This function supports only f32 and f64.
    pub fn matmul(&mut self, a: &'a Tensor<'a, F>, b: &'a Tensor<'a, F>) -> &'a Tensor<'a, F> {
        let op = dot_ops::MatMul {
            transpose_a: false,
            transpose_b: false,
        };
        Tensor::builder()
            .set_inputs(&[a, b])
            .build(self, op)
    }

    /// Matrix multiplication with inputs's transposition.
    ///
    /// Similar specification as `matmul` but, if `transpose_a` is true, `a` is transposed
    /// before actual matrix multiplication. It is the same for `transpose_b`.
    ///
    /// The performance is better than explicitly computing like `ag::matmul(ag::transpose)`.
    ///
    /// ```
    /// use autograd as ag;
    ///
    /// let ref a: ag::Tensor<f32> = ag::zeros(&[2, 4]);
    /// let ref b: ag::Tensor<f32> = ag::zeros(&[2, 3]);
    /// let ref c = ag::matmul_t(a, b, true, false);
    ///
    /// assert_eq!(c.eval(&[]).unwrap().shape(), &[4, 3]);
    /// ```
    ///
    /// This function supports only f32 and f64.
    pub fn matmul_t(
        &mut self,
        a: &'a Tensor<'a, F>,
        b: &'a Tensor<'a, F>, 
        transpose_a: bool,
        transpose_b: bool,
    ) -> &'a Tensor<'a, F> {
        let op = dot_ops::MatMul {
            transpose_a,
            transpose_b,
        };
        Tensor::builder()
            .set_inputs(&[a, b])
            .build(self, op)
    }

    /// Computes tensor-dot-product (tensor contraction) along specified axes.
    ///
    /// # Arguments
    /// * `a` - First input tensor
    /// * `b` - Second input tensor
    /// * `a_axes` - `a`'s Contraction axes
    /// * `b_axes` - `b`'s Contraction axes
    ///
    /// NOTE:
    ///
    /// * length of `a_axes` and `b_axes` must match.
    /// * Each axis number can be negative.
    /// * Supports only f32 and f64.
    ///
    /// ```
    /// use autograd as ag;
    ///
    /// let ref a: ag::Tensor<f32> = ag::zeros(&[3, 4, 5]);
    /// let ref b: ag::Tensor<f32> = ag::zeros(&[4, 3, 2]);
    /// let ref c = ag::tensordot(a, b, &[1, 0], &[0, 1]);
    /// assert_eq!(c.eval(&[]).unwrap().shape(), &[5, 2]);
    /// ```
    ///
    /// For detailed description,
    /// see https://docs.scipy.org/doc/numpy/reference/generated/numpy.tensordot.html.
    pub fn tensordot<A, B, AL>(&mut self, a: &'a Tensor<'a, F>, b: &'a Tensor<'a, F>, a_axes: &AL, b_axes: &AL) -> &'a Tensor<'a, F>
    where
        AL: ArrayLike<'a, F>,
    {
        // Preprocess
        let pre = Tensor::builder()
            .set_inputs(&[
                a,
                b,
                a_axes.as_tensor(self),
                b_axes.as_tensor(self),
            ])
            .build(self, dot_ops::TensordotPreprocess);

        let final_shape = self.nth_tensor(&pre, 0);
        let perm_a = self.nth_tensor(&pre, 1);
        let perm_b = self.nth_tensor(&pre, 2);
        let new_shape_a = self.nth_tensor(&pre, 3);
        let new_shape_b = self.nth_tensor(&pre, 4);

        let a_reshaped = self.reshape(self.transpose(a, &perm_a), &new_shape_a);
        let b_reshaped = self.reshape(self.transpose(b, &perm_b), &new_shape_b);

        // matmul
        let mm = self.matmul(&a_reshaped, &b_reshaped);
        self.reshape(mm, &final_shape)
    }

    /// Batched matrix multiplication with inputs's transposition.
    ///
    /// The rank of `a` and `b` must be equals.
    ///
    /// ```
    /// use autograd as ag;
    ///
    /// let ref a: ag::Tensor<f32> = ag::zeros(&[2, 3, 2, 4]);
    /// let ref b: ag::Tensor<f32> = ag::zeros(&[2, 3, 2, 3]);
    /// let ref c = ag::batch_matmul_t(a, b, true, false);
    ///
    /// assert_eq!(c.eval(&[]).unwrap().shape(), &[2, 3, 4, 3]);
    /// ```
    ///
    /// This function supports only f32 and f64.
    /// For detailed description, see https://www.tensorflow.org/api_docs/python/tf/matmul
    pub fn batch_matmul_t(&mut self, a: &'a Tensor<'a, F>, b: &'a Tensor<'a, F>, trans_a: bool, trans_b: bool) -> &'a Tensor<'a, F>
    {
        let op = dot_ops::BatchMatMul {
            transpose_a: trans_a,
            transpose_b: trans_b,
        };
        Tensor::builder()
            .set_inputs(&[a, b])
            .build(self, op)
    }

    /// Batched matrix multiplication.
    ///
    /// The rank of `a` and `b` must be equals.
    ///
    /// ```
    /// use autograd as ag;
    ///
    /// let ref a: ag::Tensor<f32> = ag::zeros(&[2, 3, 4, 2]);
    /// let ref b: ag::Tensor<f32> = ag::zeros(&[2, 3, 2, 3]);
    /// let ref c = ag::batch_matmul(a, b);
    ///
    /// assert_eq!(c.eval(&[]).unwrap().shape(), &[2, 3, 4, 3]);
    /// ```
    ///
    /// This function supports only f32 and f64.
    /// For detailed description, see https://www.tensorflow.org/api_docs/python/tf/matmul
    pub fn batch_matmul(&mut self, a: &'a Tensor<'a, F>, b: &'a Tensor<'a, F>) -> &'a Tensor<'a, F> {
        let op = dot_ops::BatchMatMul {
            transpose_a: false,
            transpose_b: false,
        };
        Tensor::builder()
            .set_inputs(&[a, b])
            .build(self, op)
    }

    /// Takes diff between two tensors.
    ///
    /// Returns the sorted, unique values in `a` that are not in `b`.
    ///
    /// ```
    /// use ndarray;
    /// use autograd as ag;
    ///
    /// let ref a = ag::constant(ndarray::arr1(&[4., 1., 5., 2., 3., 6.]));
    /// let ref b = ag::constant(ndarray::arr2(&[[2., 3.], [1., 4.]]));
    /// let ref c = ag::setdiff1d(a, b);
    ///
    /// assert_eq!(
    ///     c.eval(&[]),
    ///     Some(ndarray::arr1(&[5., 6.]).into_dyn())
    /// )
    /// ```
    ///
    pub fn setdiff1d(&mut self, a: &'a Tensor<'a, F>, b: &'a Tensor<'a, F>) -> &'a Tensor<'a, F> {
        let op = array_ops::SetDiff1D;
        Tensor::builder()
            .set_inputs(&[a, b])
            .build(self, op)
    }

    /// Permutes dimensions.
    ///
    /// It's like TensorFlow or NumPy's.
    ///
    /// ```
    /// use autograd as ag;
    ///
    /// let ref a: ag::Tensor<f32> = ag::zeros(&[1, 2, 3, 4, 5]);
    /// let ref b = ag::transpose(a, &[4, 2, 3, 0, 1]);
    ///
    /// assert_eq!(b.eval(&[]).unwrap().shape(), &[5, 3, 4, 1, 2]);
    /// ```
    pub fn transpose<AL: ArrayLike<'a, F>>(&mut self, x: &'a Tensor<'a, F>, perm: &AL) -> &'a Tensor<'a, F> {
        let op = math_ops::Transpose { invert_axes: false };
        Tensor::builder()
            .set_inputs(&[x, perm.as_tensor(self)])
            .build(self, op)
    }

    /// Splits input tensors into parts.
    ///
    /// Splits `x` into `sizes.len()` parts along `axis`.
    ///
    /// The size of dimension of each part is `sizes[i]` on `axis`, but is
    /// `x.shape[i]` on other axis (similar to TensorFlow's `split`).
    ///
    /// ```
    /// use autograd as ag;
    ///
    /// let ref a: ag::Tensor<f32> = ag::zeros(&[3, 7, 5]);
    /// let ref b = ag::split(a, &[2, 3, 2], 1);
    ///
    /// let evaluated = ag::eval(&[&b[0], &b[1], &b[2]], &[]);
    /// let e0 = &evaluated[0];
    /// let e1 = &evaluated[1];
    /// let e2 = &evaluated[2];
    ///
    /// assert_eq!(e0.unwrap().shape(), &[3, 2, 5]);
    /// assert_eq!(e1.unwrap().shape(), &[3, 3, 5]);
    /// assert_eq!(e2.unwrap().shape(), &[3, 2, 5]);
    /// ```
    pub fn split(&mut self, x: &'a Tensor<'a, F>, sizes: &[usize], axis: isize) -> Vec<&'a Tensor<'a, F>> {
        (0..sizes.len())
            .map(|i| {
                let start_index = sizes[..i].iter().cloned().sum::<usize>() as isize;
                let end_index = start_index + sizes[i] as isize;
                let op = array_ops::Split {
                    start_index,
                    end_index,
                    axis,
                };
                Tensor::builder().set_input(x).build(self, op)
            })
            .collect::<Vec<_>>()
    }

    /// Slices the input tensor.
    ///
    /// # Arguments
    /// * `x` - Tensor with arbitrary shape.
    /// * `starts` - Start indices for each dimensions
    /// * `ends` - End indices for each dimensions.
    /// `-1` representing the last index is acceptable for each dimension.
    ///
    /// ```
    /// use autograd as ag;
    ///
    /// let ref a: ag::Tensor<f32> = ag::zeros(&[4, 4]);
    /// let ref b = ag::slice(a, &[0, 0], &[-1, 2]); // numpy equivalent is a[:, 0:2]
    ///
    /// assert_eq!(b.eval(&[]).unwrap().shape(), &[4, 2]);
    /// ```
    pub fn slice(&mut self, x: &'a Tensor<'a, F>, starts: &[isize], ends: &[isize]) -> &'a Tensor<'a, F> {
        // TODO: Make starts and ends ArrayLike
        assert_eq!(starts.len(), ends.len());
        let starts_ends = starts.iter().zip(ends.iter());

        let indices = starts_ends
            .map(|(s, e)| {
                let slice = ndarray::Slice::new(*s, if *e == -1 { None } else { Some(*e) }, 1);
                ndarray::SliceOrIndex::from(slice)
            })
            .collect::<Vec<ndarray::SliceOrIndex>>();

        Tensor::builder()
            .set_input(x)
            .build(self, array_ops::Slice { indices })
    }

    /// Concatenates input tensors along specified axis.
    ///
    /// `axis` can be negative.
    ///
    /// ```
    /// use autograd as ag;
    ///
    /// let ref a: ag::Tensor<f32> = ag::zeros(&[3, 2]);
    /// let ref b: ag::Tensor<f32> = ag::zeros(&[3, 2]);
    /// let ref c: ag::Tensor<f32> = ag::zeros(&[3, 2]);
    /// let ref d = ag::concat(&[a, b, c], 0);
    ///
    /// assert_eq!(d.eval(&[]).unwrap().shape(), &[9, 2]);
    /// ```
    pub fn concat(&mut self, tensors: &[&'a Tensor<'a, F>], axis: isize) -> &'a Tensor<'a, F> {
        let op = array_ops::Concat { axis };
        Tensor::builder().set_inputs(tensors).build(self, op)
    }

    /// Gathers subviews from the input tensor.
    ///
    /// Same spec as https://www.tensorflow.org/api_docs/python/tf/gather.
    /// For example, this can be used for embedding vectors lookup etc.
    ///
    /// Unlike `ag::gather`, `indices` can contain negative elements.
    ///
    /// # Returns
    /// Tensor with shape `param.shape[..axis] + indices.shape + param.shape[axis+1..]`
    ///
    /// ```
    /// use ndarray;
    /// use autograd as ag;
    ///
    /// let ref param = ag::constant(ag::ndarray_ext::zeros::<f32>(&[5, 4, 8, 2]));
    /// let ref indices = ag::constant(ndarray::arr2(&[[5., -1., 3.], [2., 1., -2.]]));
    /// let ref y = ag::gather_common(param, indices, 2);
    ///
    /// assert_eq!(y.eval(&[]).unwrap().shape(), &[5, 4, 2, 3, 2])
    /// ```
    pub fn gather_common<AL: ArrayLike<'a, F>>(
        &mut self,
        param: &'a Tensor<'a, F>,
        indices: &AL,
        axis: isize,
    ) -> &'a Tensor<'a, F> {
        let op = array_ops::Gather {
            axis,
            should_normalize_negative_indices: true,
        };
        Tensor::builder()
            .set_inputs(&[&indices.as_tensor(self), param])
            .build(self, op)
    }

    /// Gathers subviews from the input tensor.
    ///
    /// Same spec as https://www.tensorflow.org/api_docs/python/tf/gather.
    /// For example, this can be used for embedding vectors lookup etc.
    ///
    /// # Returns
    /// Tensor with shape `param.shape[..axis] + indices.shape + param.shape[axis+1..]`
    ///
    /// ```
    /// use ndarray;
    /// use autograd as ag;
    ///
    /// let ref param = ag::constant(ag::ndarray_ext::zeros::<f32>(&[5, 4, 8, 2]));
    /// let ref indices = ag::constant(ndarray::arr2(&[[5., 4., 3.], [2., 1., 0.]]));  // shape: (2, 3)
    /// let ref y = ag::gather(param, indices, 2);
    ///
    /// assert_eq!(y.eval(&[]).unwrap().shape(), &[5, 4, 2, 3, 2])  // [5, 4] + [2, 3] + [2g
    /// ```
    pub fn gather<AL, A>(&mut self, param: &'a Tensor<'a, F>, indices: &AL, axis: isize) -> &'a Tensor<'a, F>
    where
        AL: ArrayLike<'a, F>,
        A: AsRef<Tensor<'a, F>>,
    {
        let op = array_ops::Gather {
            axis,
            should_normalize_negative_indices: false,
        };
        Tensor::builder()
            .set_inputs(&[&indices.as_tensor(self), param])
            .build(self, op)
    }

    /// Normalizes the input tensor with its mean and variance along specified axis.
    ///
    /// ```
    /// use ndarray;
    /// use autograd as ag;
    ///
    /// let ref x: ag::Tensor<f32> = ag::standard_normal(&[3, 4]);
    /// let ref y1 = ag::normalize(x, &[0]);
    /// let ref y2 = ag::normalize(x, &[0]);
    ///
    /// let evaluated = ag::eval(&[y1, y2], &[]);
    /// let e0 = &evaluated[0];
    /// let e1 = &evaluated[1];
    /// assert_eq!(e0.unwrap().shape(), &[3, 4]);
    /// assert_eq!(e1.unwrap().shape(), &[3, 4]);
    /// ```
    pub fn normalize<AL: ArrayLike<'a, F>>(&mut self, x: &'a Tensor<'a, F>, axes: &AL) -> &'a Tensor<'a, F> {
        let x = x;
        let axes = axes.as_tensor(self);
        let mean = self.reduce_mean(x, axes, true);
        let centered = x - mean;
        let variance = self.reduce_mean(self.square(centered), axes, true);
        let em5 = F::from(1e-5).unwrap();
        (x - mean) / self.sqrt(variance + em5)
    }

    /// Applies batch normalization.
    ///
    /// `scale` and `shift` should be shared variables.
    /// Since normalization is performed along 1st axis of `x`,
    /// both of them should have shape `(1, x.shape[1])`
    ///
    /// ```
    /// use ndarray;
    /// use autograd as ag;
    ///
    /// let ref x = ag::standard_normal(&[3, 4]);
    /// let ref scale = ag::variable(ag::ndarray_ext::ones::<f32>(&[1, 4]));
    /// let ref shift = ag::variable(ag::ndarray_ext::zeros::<f32>(&[1, 4]));
    /// let ref norm = ag::batch_norm(x, scale, shift);
    ///
    /// assert_eq!(norm.eval(&[]).unwrap().shape(), &[3, 4]);
    /// ```
    pub fn batch_norm(&mut self, x: &'a Tensor<'a, F>, scale: &'a Tensor<'a, F>, shift: &'a Tensor<'a, F>) -> &'a Tensor<'a, F>
    {
        self.normalize(x, &[0]) * scale + shift
    }

    /// Generates a zero-ranked tensor from a scalar value.
    ///
    /// ```
    /// use autograd as ag;
    ///
    /// let a = ag::scalar(3.);
    /// println!("{}", a.eval(&[]).unwrap());  // => 3.
    /// assert_eq!(a.eval(&[]).unwrap().shape(), &[]);
    /// ```
    pub fn scalar(&mut self, val: F) -> &'a Tensor<'a, F> {
        let op = const_gen_ops::Scalar { val };
        Tensor::builder()
            .set_shape(self.convert_to_tensor(crate::ndarray_ext::scalar_shape()))
            .build(self, op)
    }

    /// Outputs values sampled from the normal distribution.
    pub fn random_normal<AL: ArrayLike<'a, F>>(&mut self, shape: &AL, mean: f64, stddev: f64) -> &'a Tensor<'a, F> {
        self.random_normal_rng(Default::default(), shape, mean, stddev)
    }

    /// Outputs values sampled from the normal distribution.
    pub fn random_normal_rng<AL: ArrayLike<'a, F>, R: Rng + Send + 'static>(
        &mut self,
        arr_rng: ArrRng<F, R>,
        shape: &AL,
        mean: f64,
        stddev: f64,
    ) -> &'a Tensor<'a, F> {
        let shape = shape.as_tensor(self);
        Tensor::builder()
            .set_input(&shape)
            .set_shape(shape)
            .build(self, random_ops::RandomNormal::new(arr_rng, mean, stddev))
    }

    /// Outputs values sampled from the uniform distribution.
    pub fn random_uniform<AL: ArrayLike<'a, F>>(&mut self, shape: &AL, min: f64, max: f64) -> &'a Tensor<'a, F> {
        self.random_uniform_rng(Default::default(), shape, min, max)
    }

    /// Outputs values sampled from the uniform distribution.
    ///
    /// See https://github.com/raskr/rust-autograd/issues/1.
    pub fn random_uniform_rng<AL: ArrayLike<'a, F>, R: Rng + Send + 'static>(
        &mut self,
        arr_rng: ArrRng<F, R>,
        shape: &AL,
        min: f64,
        max: f64,
    ) -> &'a Tensor<'a, F> {
        let shape = shape.as_tensor(self);
        Tensor::builder()
            .set_input(&shape)
            .set_shape(shape)
            .build(self, random_ops::RandomUniform::new(arr_rng, min, max))
    }

    /// Outputs values sampled from the standard normal distribution.
    pub fn standard_normal<AL: ArrayLike<'a, F>>(&mut self, shape: &AL) -> &'a Tensor<'a, F> {
        self.standard_normal_rng(Default::default(), shape)
    }

    /// Outputs values sampled from the standard normal distribution.
    ///
    /// See https://github.com/raskr/rust-autograd/issues/1.
    pub fn standard_normal_rng<AL: ArrayLike<'a, F>, R: Rng + Send + 'static>(
        &mut self,
        arr_rng: ArrRng<F, R>,
        shape: &AL,
    ) -> &'a Tensor<'a, F> {
        let shape = shape.as_tensor(self);
        Tensor::builder()
            .set_input(&shape)
            .set_shape(shape)
            .build(self, random_ops::StandardNormal::new(arr_rng))
    }

    /// Outputs values sampled from the standard uniform distribution.
    pub fn standard_uniform<AL: ArrayLike<'a, F>>(&mut self, shape: &AL) -> &'a Tensor<'a, F> {
        self.standard_uniform_rng(Default::default(), shape)
    }

    /// Outputs values sampled from the standard uniform distribution.
    ///
    /// See https://github.com/raskr/rust-autograd/issues/1.
    pub fn standard_uniform_rng<AL: ArrayLike<'a, F>, R: Rng + Send + 'static>(
        &mut self,
        arr_rng: ArrRng<F, R>,
        shape: &AL,
    ) -> &'a Tensor<'a, F> {
        let shape = shape.as_tensor(self);
        Tensor::builder()
            .set_input(&shape)
            .set_shape(shape)
            .build(self, random_ops::StandardUniform::new(arr_rng))
    }

    /// Outputs values sampled from the bernoulli distribution.
    pub fn bernoulli<AL: ArrayLike<'a, F>>(&mut self, shape: &AL, p: f64) -> &'a Tensor<'a, F> {
        self.bernoulli_rng(Default::default(), shape, p)
    }

    /// Outputs values sampled from the bernoulli distribution.
    ///
    /// See https://github.com/raskr/rust-autograd/issues/1.
    pub fn bernoulli_rng<AL: ArrayLike<'a, F>, R: Rng + Send + 'static>(
        &mut self,
        arr_rng: ArrRng<F, R>,
        shape: &AL,
        p: f64,
    ) -> &'a Tensor<'a, F> {
        let shape = shape.as_tensor(self);
        Tensor::builder()
            .set_input(&shape)
            .set_shape(shape)
            .build(self, random_ops::Bernoulli::new(arr_rng, p))
    }

    /// Outputs values sampled from the exponential distribution.
    pub fn random_exp<AL: ArrayLike<'a, F>>(&mut self, shape: &AL, lambda: f64) -> &'a Tensor<'a, F> {
        self.random_exp_rng(Default::default(), shape, lambda)
    }

    /// Outputs values sampled from the exponential distribution.
    ///
    /// See https://github.com/raskr/rust-autograd/issues/1.
    pub fn random_exp_rng<AL: ArrayLike<'a, F>, R: Rng + Send + 'static>(
        &mut self,
        arr_rng: ArrRng<F, R>,
        shape: &AL,
        lambda: f64,
    ) -> &'a Tensor<'a, F> {
        let shape = shape.as_tensor(self);
        Tensor::builder()
            .set_input(&shape)
            .set_shape(shape)
            .build(self, random_ops::Exponential::new(arr_rng, lambda))
    }

    /// Outputs values sampled from the gamma distribution.
    pub fn random_gamma<AL: ArrayLike<'a, F>>(
        &mut self,
        shape: &AL,
        shape_param: f64,
        scale: f64,
    ) -> &'a Tensor<'a, F> {
        self.random_gamma_rng(Default::default(), shape, shape_param, scale)
    }

    /// Outputs values sampled from the gamma distribution.
    ///
    /// See https://github.com/raskr/rust-autograd/issues/1.
    pub fn random_gamma_rng<AL: ArrayLike<'a, F>, R: Rng + Send + 'static>(
        &mut self,
        arr_rng: ArrRng<F, R>,
        shape: &AL,
        shape_param: f64,
        scale: f64,
    ) -> &'a Tensor<'a, F> {
        let shape = shape.as_tensor(self);
        Tensor::builder()
            .set_input(&shape)
            .set_shape(shape)
            .build(self, random_ops::Gamma::new(arr_rng, shape_param, scale))
    }

    /// Outputs values sampled from the log-normal distribution.
    pub fn log_normal<AL: ArrayLike<'a, F>>(&mut self, shape: &AL, mean: f64, stddev: f64) -> &'a Tensor<'a, F> {
        self.log_normal_rng(Default::default(), shape, mean, stddev)
    }

    /// Outputs values sampled from the log-normal distribution.
    ///
    /// See https://github.com/raskr/rust-autograd/issues/1.
    pub fn log_normal_rng<AL: ArrayLike<'a, F>, R: Rng + Send + 'static>(
        &mut self,
        arr_rng: ArrRng<F, R>,
        shape: &AL,
        mean: f64,
        stddev: f64,
    ) -> &'a Tensor<'a, F> {
        let shape = shape.as_tensor(self);
        Tensor::builder()
            .set_input(&shape)
            .set_shape(shape)
            .build(self, random_ops::LogNormal::new(arr_rng, mean, stddev))
    }

    /// Converts an `ndarray::Array` to a `ag::Tensor`.
    ///
    /// ```
    /// use ndarray;
    /// use autograd as ag;
    ///
    /// let arr = ndarray::arr1(&[2., 3.]);
    /// let tensor = ag::convert_to_tensor(arr.clone());
    /// assert_eq!(tensor.eval(&[]), Some(arr.into_dyn()));
    /// ```
    pub fn convert_to_tensor<D>(&mut self, arr: ndarray::Array<F, D>) -> &'a Tensor<'a, F>
    where
        D: ndarray::Dimension,
    {
        let arr = arr.into_dyn();
        let shape = {
            let op = const_gen_ops::ConvertToTensor {
                arr: crate::ndarray_ext::shape_of(&arr),
            };
            Tensor::builder().build(self, op)
        };
        Tensor::builder()
            .set_shape(shape)
            .build(self, const_gen_ops::ConvertToTensor { arr })
    }

    /// Returns zeros with given shape.
    ///
    /// ```
    /// use ndarray;
    /// use autograd as ag;
    ///
    /// let a: ag::Tensor<f32> = ag::zeros(&[4, 2]);
    /// assert_eq!(a.eval(&[]), Some(ndarray::Array2::<f32>::zeros((4, 2)).into_dyn()));
    /// ```
    pub fn zeros<AL: ArrayLike<'a, F>>(&mut self, shape: &AL) -> &'a Tensor<'a, F> {
        Tensor::builder()
            .set_input(shape.as_tensor(self))
            .build(self, const_gen_ops::Zeros)
    }

    /// Returns ones with given shape.
    ///
    /// ```
    /// use ndarray;
    /// use autograd as ag;
    ///
    /// let a = ag::ones(&[4, 2]);
    /// assert_eq!(a.eval(&[]), Some(ndarray::Array2::<f32>::from_elem((4, 2), 1.).into_dyn()));
    /// ```
    pub fn ones<AL: ArrayLike<'a, F>>(&mut self, shape: &AL) -> &'a Tensor<'a, F> {
        Tensor::builder()
            .set_input(shape.as_tensor(self))
            .build(self, const_gen_ops::Ones)
    }

    /// Returns a range.
    ///
    /// ```
    /// use ndarray;
    /// use autograd as ag;
    ///
    /// let start = 0.;
    /// let end = 5.;
    /// let step = 1.;
    /// let ref z = ag::range(start, end, step);
    ///
    /// assert_eq!(z.eval(&[]), Some(ndarray::Array1::range(0., 5., 1.).into_dyn()));
    /// ```
    pub fn range(&mut self, start: F, end: F, step: F) -> &'a Tensor<'a, F> {
        Tensor::builder()
            .set_inputs(&[self.scalar(start), self.scalar(end), self.scalar(step)])
            .build(self, const_gen_ops::Range)
    }

    pub fn _range<AL: ArrayLike<'a, F>>(&mut self, start: &AL, end: &AL, step: &AL) -> &'a Tensor<'a, F> {
        Tensor::builder()
            .set_inputs(&[&start.as_tensor(self), end.as_tensor(self), step.as_tensor(self)])
            .build(self, const_gen_ops::Range)
    }

    /// 2D convolution.
    ///
    /// * `x`: Tensor with shape `(batch, channel, h, w)`
    /// * `w`: Tensor with shape `(out_channel, channel, filter_h, filter_w)`
    ///
    /// Returns a tensor with shape `(batch, out_channel, out_h, out_w)`
    ///
    /// where
    ///
    ///   * `out_h` = `(h + 2 * pad - filter_h) / stride + 1`
    ///   * `out_w` = `(w + 2 * pad - filter_w) / stride + 1`
    ///
    /// This function supports only f32 and f64.
    pub fn conv2d(&mut self, x: &'a Tensor<'a, F>, w: &'a Tensor<'a, F>, pad: usize, stride: usize) -> &'a Tensor<'a, F>
    {
        Tensor::builder()
            .set_inputs(&[x, w])
            .build(self, conv_ops::conv2d::Conv2D {
                pad,
                stride,
                dilation: 1,
            })
    }

    /// 2D convolution with dilation.
    ///
    /// * `x`: Tensor with shape `(batch, channel, h, w)`
    /// * `w`: Tensor with shape `(out_channel, in_channel, filter_h, filter_w)`
    ///
    /// Returns a tensor with shape `(batch, out_channel, out_h, out_w)`
    ///
    /// where
    ///
    ///   * `out_h` = `(h + 2 * pad - (dilate * (filter - 1) + 1)) / stride + 1`
    ///   * `out_w` = `(w + 2 * pad - (dilate * (filter - 1) + 1)) / stride + 1`
    ///
    /// This function supports only f32 and f64.
    pub fn dilated_conv2d(&mut self, x: &'a Tensor<'a, F>, w: &'a Tensor<'a, F>, pad: usize, stride: usize, dilate: usize) -> &'a Tensor<'a, F>
    {
        Tensor::builder()
            .set_inputs(&[x, w])
            .build(self, conv_ops::conv2d::Conv2D {
                pad,
                stride,
                dilation: dilate,
            })
    }

    /// 2D transposed convolution.
    ///
    /// * `x`: Tensor with shape `(batch, in_channel, h, w)`
    /// * `w`: Tensor with shape `(in_channel, out_channel, filter_h, filter_w)`
    ///
    /// Returns a tensor with shape `(batch, out_channel, out_h, out_w)`
    ///
    /// where
    ///
    ///   * `out_h` = `stride * (h - 1) - pad + filter_h`
    ///   * `out_w` = `stride * (w - 1) - pad + filter_w`
    ///
    /// This function supports only f32 and f64.
    pub fn conv2d_transpose(&mut self, x: &'a Tensor<'a, F>, w: &'a Tensor<'a, F>, pad: usize, stride: usize) -> &'a Tensor<'a, F>
    {
        Tensor::builder()
            .set_inputs(&[x, w])
            .build(self, conv_ops::conv2d_transpose::Conv2DTranspose {
                pad,
                stride,
                dilation: 1,
            })
    }

    /// 2D transposed convolution with dilation.
    ///
    /// * `x`: Tensor with shape `(batch, in_channel, h, w)`
    /// * `w`: Tensor with shape `(in_channel, out_channel, filter_h, filter_w)`
    ///
    /// Returns a tensor with shape `(batch, out_channel, out_h, out_w)`
    ///
    /// where
    ///
    ///   * `out_h` = `stride * (h - 1) - pad + (dilate * (filter_h - 1) + 1)`
    ///   * `out_w` = `stride * (w - 1) - pad + (dilate * (filter_w - 1) + 1)`
    ///
    /// This function supports only f32 and f64.
    pub fn dilated_conv2d_transpose(
        &mut self,
        x: &'a Tensor<'a, F>,
        w: &'a Tensor<'a, F>,
        pad: usize,
        stride: usize,
        dilate: usize,
    ) -> &'a Tensor<'a, F>
    {
        Tensor::builder()
            .set_inputs(&[x, w])
            .build(self, conv_ops::conv2d_transpose::Conv2DTranspose {
                pad,
                stride,
                dilation: dilate,
            })
    }

    /// 2D max pooling.
    ///
    /// * `x`: Tensor with shape `(batch, channel, h, w)`
    ///
    /// Returns a tensor with shape `(batch, channel, out_h, out_w)`
    ///
    /// where
    ///
    ///   * `out_h` = `(h + 2 * pad - pool_size) / stride + 1`
    ///   * `out_w` = `(w + 2 * pad - pool_size) / stride + 1`
    ///
    /// This function supports only f32 and f64.
    pub fn max_pool2d(
        &mut self,
        x: &'a Tensor<'a, F>,
        pool_size: usize,
        pad: usize,
        stride: usize,
    ) -> &'a Tensor<'a, F> {
        Tensor::builder()
            .set_input(x)
            .build(self, conv_ops::max_pool2d::MaxPool2D {
                pad,
                stride,
                size: pool_size,
            })
    }
}
