//! A collection of functions to manipulate `ag::Tensor` objects
use ndarray;

use crate::ndarray_ext::{ArrRng, NdArray};
use crate::scope::Scope;
use crate::tensor::{Tensor, ScopedTensor};
use crate::Float;
use rand::Rng;

mod activation_ops;
mod array_ops;
mod basic_source_ops;
#[doc(hidden)]
pub mod binary_ops;
mod const_gen_ops;
mod conv_ops;
pub mod dot_ops;
pub mod gradient_descent_ops;
mod gradient_ops;
pub mod hook_ops;
mod math_ops;
mod random_ops;
mod reduction_ops;
mod xent_ops;

// ---------------------------------------
// -- Ops to manipulate `Tensor` object --
// ---------------------------------------

impl<'a, 'b: 'a, F: Float> ScopedTensor<'a, 'b, F> {
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
    pub fn get(&'b self, i: isize, c: &'b Scope<F>) -> ScopedTensor<'a, 'b, F> {
        let op = array_ops::IndexOp { index: i };
        Tensor::builder().set_input(self).build(c, op)
    }
}

impl<'a, 'b: 'a, 'r: 'a, 'l: 'a, F: Float> crate::scope::Scope<F> {
    
    #[inline]
    pub(crate) fn scope(&'b self, x: &'a Tensor<F>) -> ScopedTensor<'a, 'b, F> {
        ScopedTensor {
            inner: x, 
            scope: self
        }
    }
    
    #[inline]
    pub fn newshape(&'b self, slice: &[isize]) -> ScopedTensor<'a, 'b, F> {
        self.isize_slice_to_tensor(slice)
    }

    #[inline]
    pub fn axes(&'b self, slice: &[isize]) -> ScopedTensor<'a, 'b, F> {
        self.isize_slice_to_tensor(slice)
    }

    fn isize_slice_to_tensor(&'b self, slice: &[isize]) -> ScopedTensor<'a, 'b, F> {
        let vec = slice
            .iter()
            .map(|&a| F::from(a).unwrap())
            .collect::<Vec<F>>();
        // unwrap is safe
        let arr = NdArray::from_shape_vec(ndarray::IxDyn(&[slice.len()]), vec).unwrap();
        self.convert_to_tensor(arr)
    }

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
    pub fn grad<A, B>(&'b self, ys_: &[A], xs: &[B]) -> Vec<ScopedTensor<'a, 'b, F>>
    where
        A: AsRef<ScopedTensor<'a, 'b, F>>,
        B: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        let len = ys_.len();
        let mut ys = Vec::with_capacity(len);
        for y in ys_ {
            ys.push(self.reduce_sum_to_scalar(y));
        }
        let gys = vec![self.scalar(F::one()); len];
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
    pub unsafe fn grad_with_default<A, B, C>(
        &'b self,
        ys: &[A],
        xs: &[B],
        ys_grads: &[C],
    ) -> Vec<ScopedTensor<'a, 'b, F>>
    where
        A: AsRef<ScopedTensor<'a, 'b, F>>,
        B: AsRef<ScopedTensor<'a, 'b, F>>,
        C: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        let xs: Vec<_> = xs.iter().map(|x| x.as_ref().inner).collect();
        let ys: Vec<_> = ys.iter().map(|y| y.as_ref().inner).collect();
        let ys_grads: Vec<_> = ys_grads.iter().map(|x| x.as_ref().inner).collect();
        crate::gradient::symbolic_gradients(ys.as_slice(), xs.as_slice(), ys_grads.as_slice(), self)
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
    pub fn jacobians<A: 'a, B: 'a>(
        &'b self,
        y_: A,
        xs_: &[B],
        objective_len: usize,
    ) -> Vec<ScopedTensor<'a, 'b, F>>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
            B: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        let y = y_.as_ref();
        let xs: Vec<_> = xs_.iter().map(|x| x.as_ref().inner).collect();
        let mut vec_vec = Vec::with_capacity(objective_len);
        for i in 0..objective_len as isize {
            vec_vec.push({
                crate::gradient::symbolic_gradients(&[&y.get(i, self)], xs.as_slice(), &[], self)
            });
        }

        let len = xs.len();
        let mut ret = Vec::with_capacity(len);
        // post process gradients
        for i in 0..len {
            // jac is matrix
            let mut jac = Vec::with_capacity(objective_len);
            for j in 0..objective_len {
                jac.push(self.expand_dims(self.flatten(&vec_vec[j][i]), self.axes(&[0])));
            }
            // (y size, x size)
            ret.push(self.concat(&jac, 0));
        }
        ret
    }

//    /// (Experimental) Computes hessian vector product
//    pub fn _hessian_vector_product<A, B, C: 'a>(
//        &'b self,
//        ys: &[A],
//        xs: &[B],
//        vectors: &[C],
//    ) -> Vec<ScopedTensor<'a, 'b, F>>
//        where
//            A: AsRef<ScopedTensor<'a, 'b, F>>,
//            B: AsRef<ScopedTensor<'a, 'b, F>>,
//            C: AsRef<ScopedTensor<'a, 'b, F>>,
//    {
//        let grads = self.grad(ys, xs);
//        let products = grads
//            .into_iter()
//            .zip(vectors)
//            .map(|(g, &v)| g.as_ref() * v.as_ref())
//            .collect::<Vec<_>>();
//        self.grad(products.as_slice(), xs)
//    }

    /// Stops gradient propagation.
    ///
    /// Guarantees that the gradient is not propagated to the tensors behind this
    /// during gradient computation.
    pub fn stop_gradient<A>(&'b self, x: A) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        Tensor::builder()
            .set_input(x.as_ref())
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
    pub fn variable<D: ndarray::Dimension>(&'b self, arr: ndarray::Array<F, D>) -> ScopedTensor<'a, 'b, F> {
        let arr = arr.into_dyn();
        Tensor::builder()
            .set_shape(self.convert_to_tensor(crate::ndarray_ext::shape_of(&arr)).inner)
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
    pub fn placeholder(&'b self, shape_: &[isize]) -> ScopedTensor<'a, 'b, F> {
        let b = Tensor::builder().set_is_placeholder(true);
        let rank = shape_.len();
        let b = if rank == 0 || -1 != shape_[0] {
            b.set_shape(
                self.convert_to_tensor(
                    NdArray::from_shape_vec(
                        ndarray::IxDyn(&[rank]),
                        shape_
                            .iter()
                            .map(|&x| F::from(x).unwrap())
                            .collect::<Vec<_>>(),
                    )
                        .unwrap(),
                ).inner,
            )
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
    pub fn constant<D>(&'b self, arr: ndarray::Array<F, D>) -> ScopedTensor<'a, 'b, F>
        where
            D: ndarray::Dimension,
    {
        let arr = arr.into_dyn();
        Tensor::builder()
            .set_shape(self.convert_to_tensor(crate::ndarray_ext::shape_of(&arr)).inner)
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
    pub fn shape<A>(&'b self, x: A) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        if let Some(ref inner) = x.as_ref().shape {
            ScopedTensor { inner: unsafe { &**inner }, scope: self }
        } else {
            Tensor::builder()
                .set_input(x.as_ref()).set_differentiable(false)
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
    pub fn size<A>(&'b self, x: A) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        Tensor::builder()
            .set_input(x.as_ref())
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
    pub fn rank<A>(&'b self, x: A) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        Tensor::builder()
            .set_input(x.as_ref()).set_differentiable(false)
            .build(self, array_ops::Rank)
    }

    /// Elementwise sine
    pub fn sin<A>(&'b self, x: A) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        Tensor::builder()
            .set_input(x.as_ref()).set_shape(&self.shape(x))
            .build(self, math_ops::Sin)
    }

    /// Elementwise cosine
    pub fn cos<A>(&'b self, x: A) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        Tensor::builder()
            .set_input(x.as_ref()).set_shape(&self.shape(x))
            .build(self, math_ops::Cos)
    }

    /// Elementwise tangent
    pub fn tan<A>(&'b self, x: A) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        Tensor::builder()
            .set_input(x.as_ref()).set_shape(&self.shape(x))
            .build(self, math_ops::Tan)
    }

    /// Elementwise arcsin
    pub fn asin<A>(&'b self, x: A) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        Tensor::builder()
            .set_input(x.as_ref()).set_shape(&self.shape(x))
            .build(self, math_ops::Asin)
    }

    /// Elementwise arccos
    pub fn acos<A>(&'b self, x: A) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        Tensor::builder()
            .set_input(x.as_ref()).set_shape(&self.shape(x))
            .build(self, math_ops::Acos)
    }

    /// Elementwise arctan
    pub fn atan<A>(&'b self, x: A) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        Tensor::builder()
            .set_input(x.as_ref()).set_shape(&self.shape(x))
            .build(self, math_ops::Atan)
    }

    /// Elementwise hyperbolic sine
    pub fn sinh<A>(&'b self, x: A) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        Tensor::builder()
            .set_input(x.as_ref()).set_shape(&self.shape(x))
            .build(self, math_ops::Sinh)
    }

    /// Elementwise hyperbolic cosine
    pub fn cosh<A>(&'b self, x: A) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        Tensor::builder()
            .set_input(x.as_ref()).set_shape(&self.shape(x))
            .build(self, math_ops::Cosh)
    }

    /// Elementwise hyperbolic tangent
    pub fn tanh<A>(&'b self, x: A) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        Tensor::builder()
            .set_input(x.as_ref()).set_shape(&self.shape(x))
            .build(self, math_ops::Tanh)
    }

    /// Elementwise hyperbolic arcsin
    pub fn asinh<A>(&'b self, x: A) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        Tensor::builder()
            .set_input(x.as_ref()).set_shape(&self.shape(x))
            .build(self, math_ops::Asinh)
    }

    /// Elementwise hyperbolic arccos
    pub fn acosh<A>(&'b self, x: A) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        Tensor::builder()
            .set_input(x.as_ref()).set_shape(&self.shape(x))
            .build(self, math_ops::Acosh)
    }

    /// Elementwise hyperbolic arctan
    pub fn atanh<A>(&'b self, x: A) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        Tensor::builder()
            .set_input(x.as_ref()).set_shape(&self.shape(x))
            .build(self, math_ops::Atanh)
    }

    #[doc(hidden)]
    /// Gets n th tensor in `x`.
    ///
    /// `x` must be a result of a multi-outputs op;
    /// otherwise index-out-of-bounds error may happen.
    pub fn nth_tensor<A>(&'b self, x: A, n: usize) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        Tensor::builder()
            .set_input(x.as_ref()).set_input_indices(vec![n])
            .build(self, activation_ops::Identity)
    }

    /// Identity function without copy.
    pub fn identity<A>(&'b self, x: A) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        Tensor::builder()
            .set_input(x.as_ref()).set_shape(&self.shape(x))
            .build(self, activation_ops::Identity)
    }

    #[inline]
    fn infer_bin_op_shape<A, B>(&'b self, shape_a: A, shape_b: B) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
            B: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        Tensor::builder()
            .set_inputs(&[shape_a.as_ref(), shape_b.as_ref()])
            .build(self, array_ops::InferBinOpShape)
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
//    pub fn add<A, B>(&'b self, a: A, b: B) -> ScopedTensor<'a, 'b, F>
    pub fn add(&'b self, a: &'a Tensor<F>, b: &'a Tensor<F>) -> ScopedTensor<'a, 'b, F>
//    where
//        A: AsRef<ScopedTensor<'a, 'b, F>> + 'l,
//        B: AsRef<ScopedTensor<'a, 'b, F>> + 'r,
    {
        let a_ = &self.scope(a);
        let b_ = &self.scope(b);
        Tensor::builder()
//            .set_shape(&self.infer_bin_op_shape(self.shape(a_), self.shape(b_)))
            .set_inputs(&[a_, b_])
            .build(self, binary_ops::AddOp)
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
    pub fn sub(&'b self, a: &'a Tensor<F>, b: &'a Tensor<F>) -> ScopedTensor<'a, 'b, F>
//    pub fn sub(&'b self, a: &ScopedTensor<'a, 'b, F>, b: &ScopedTensor<'a, 'b, F>) -> ScopedTensor<'a, 'b, F>
//    pub fn sub<A, B>(&'b self, a: A, b: B) -> ScopedTensor<'a, 'b, F>
//        where
//            A: AsRef<ScopedTensor<'a, 'b, F>>,
//            B: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        let a_ = &self.scope(a);
        let b_ = &self.scope(b);
        Tensor::builder()
//            .set_shape(&self.infer_bin_op_shape(self.shape(a_), self.shape(b_)))
            .set_inputs(&[a_, b_])
            .build(self, binary_ops::SubOp)
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
    pub fn mul(&'b self, a: &'a Tensor<F>, b: &'a Tensor<F>) -> ScopedTensor<'a, 'b, F>
//    pub fn mul(&'b self, a: &ScopedTensor<'a, 'b, F>, b: &ScopedTensor<'a, 'b, F>) -> ScopedTensor<'a, 'b, F>
//        where
//            A: AsRef<ScopedTensor<'a, 'b, F>>,
//            B: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        let a_ = &self.scope(a);
        let b_ = &self.scope(b);
        Tensor::builder()
//            .set_shape(&self.infer_bin_op_shape(self.shape(a_), self.shape(b_)))
            .set_inputs(&[a_, b_])
            .build(self, binary_ops::MulOp)
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
    pub fn div(&'b self, a: &'a Tensor<F>, b: &'a Tensor<F>) -> ScopedTensor<'a, 'b, F>
//    pub fn div(&'b self, a: &ScopedTensor<'a, 'b, F>, b: &ScopedTensor<'a, 'b, F>) -> ScopedTensor<'a, 'b, F>
//        where
//            A: AsRef<ScopedTensor<'a, 'b, F>>,
//            B: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        let a_ = &self.scope(a);
        let b_ = &self.scope(b);
        Tensor::builder()
//            .set_shape(&self.infer_bin_op_shape(self.shape(a_), self.shape(b_)))
            .set_inputs(&[a_, b_])
            .build(self, binary_ops::DivOp)
    }

    /// Elementwise sqrt
    pub fn sqrt<A>(&'b self, x: A) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        Tensor::builder()
            .set_input(x.as_ref()).set_shape(&self.shape(x))
            .build(self, math_ops::Sqrt)
    }

    /// Elementwise pow
    pub fn pow<A>(&'b self, x: A, a: F) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        Tensor::builder()
            .set_input(x.as_ref()).set_shape(&self.shape(x))
            .build(self, math_ops::Pow { a })
    }

    /// Elementwise log
    pub fn log<A>(&'b self, x: A, a: F) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        Tensor::builder()
            .set_input(x.as_ref()).set_shape(&self.shape(x))
            .build(self, math_ops::Log { a })
    }

    /// Elementwise exponential
    pub fn exp<A>(&'b self, x: A) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        Tensor::builder()
            .set_input(x.as_ref()).set_shape(&self.shape(x))
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
    pub fn maximum<A, B>(&'b self, a: A, b: B) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
            B: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        Tensor::builder()
            .set_inputs(&[a.as_ref(), b.as_ref()])
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
    pub fn minimum<A, B>(&'b self, a: A, b: B) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
            B: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        Tensor::builder()
            .set_inputs(&[a.as_ref(), b.as_ref()])
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
    pub fn add_n<A>(&'b self, xs: &'a [A]) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        let len = xs.len();
        assert_ne!(len, 0);
        if len == 1 {
            self.scope(xs[0].as_ref())
        } else {
            Tensor::builder()
                .set_inputs(xs.iter().map(|x| x.as_ref()).collect::<Vec<_>>().as_slice())
                .set_shape(&self.shape(xs[0]))
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
    pub fn equal<A, B>(&'b self, a: A, b: B) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
            B: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        Tensor::builder()
            .set_inputs(&[a.as_ref(), b.as_ref()])
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
    pub fn not_equal<A, B>(&'b self, a: A, b: B) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
            B: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        Tensor::builder()
            .set_inputs(&[a.as_ref(), b.as_ref()])
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
    pub fn argmax<A>(&'b self, x: A, axis: isize, keep_dim: bool) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        let op = reduction_ops::ArgMax { axis, keep_dim };
        Tensor::builder().set_input(x.as_ref()).build(self, op)
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
    pub fn expand_dims<A, B>(&'b self, x: A, axes: B) -> ScopedTensor<'a, 'b, F>
    where
        A: AsRef<ScopedTensor<'a, 'b, F>>,
        B: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        Tensor::builder()
            .set_inputs(&[x.as_ref(), axes.as_ref()])
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
    pub fn squeeze<A, B>(&'b self, x: A, axes: B) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
            B: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        Tensor::builder()
            .set_inputs(&[x.as_ref(), axes.as_ref()])
            .build(self, array_ops::Squeeze)
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
    pub fn tile<A>(&'b self, x: A, axis: isize, num: usize) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        let op = array_ops::Tile { axis, num };
        Tensor::builder().set_input(x.as_ref()).build(self, op)
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
    pub fn clip<A>(&'b self, x: A, min: F, max: F) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        let op = array_ops::Clip { min, max };
        Tensor::builder().set_input(x.as_ref()).build(self, op)
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
    pub fn reduce_max<A, B>(&'b self, x: A, axes: B, keep_dims: bool) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
            B: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        let op = reduction_ops::ReduceMax {
            keep_dims,
            sparse_axes: false,
        };
        Tensor::builder().set_inputs(&[x.as_ref(), axes.as_ref()]).build(self, op)
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
    pub fn reduce_min<A, B>(&'b self, x: A, axes: B, keep_dims: bool) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
            B: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        let op = reduction_ops::ReduceMin {
            keep_dims,
            sparse_axes: false,
        };
        Tensor::builder().set_inputs(&[x.as_ref(), axes.as_ref()]).build(self, op)
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
    pub fn reduce_sum_to_scalar<A>(&'b self, x: A) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        Tensor::builder()
            .set_input(x.as_ref()).build(self, reduction_ops::ReduceSumToScalar)
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
    pub fn reduce_sum<A, B>(&'b self, x: A, axes: B, keep_dims: bool) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
            B: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        let op = reduction_ops::ReduceSum {
            keep_dims,
            sparse_axes: false,
        };
        Tensor::builder().set_inputs(&[x.as_ref(), axes.as_ref()]).build(self, op)
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
    pub fn reduce_mean<A, B>(&'b self, x: A, axes: B, keep_dims: bool) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
            B: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        let op = reduction_ops::ReduceMean {
            keep_dims,
            sparse_axes: false,
        };
        Tensor::builder().set_inputs(&[x.as_ref(), axes.as_ref()]).build(self, op)
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
    pub fn reduce_prod<A, B>(&'b self, x: A, axes: B, keep_dims: bool) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
            B: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        let op = reduction_ops::ReduceProd {
            keep_dims,
            sparse_axes: false,
        };
        Tensor::builder().set_inputs(&[x.as_ref(), axes.as_ref()]).build(self, op)
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
    pub fn reshape<A, B>(&'b self, x: A, shape: B) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
            B: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        Tensor::builder()
            .set_inputs(&[x.as_ref(), shape.as_ref()])
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
    pub fn flatten<A>(&'b self, x: A) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        Tensor::builder()
            .set_inputs(&[x.as_ref(), &self.scalar(F::one().neg())])
            .set_shape(&self.shape(x))
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
    pub fn sign<A>(&'b self, a: A) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        Tensor::builder()
            .set_shape(&self.shape(a))
            .set_input(a.as_ref())
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
    pub fn abs<A>(&'b self, a: A) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        Tensor::builder()
            .set_shape(&self.shape(a))
            .set_input(a.as_ref())
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
    pub fn floor<A>(&'b self, a: A) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        Tensor::builder()
            .set_shape(&self.shape(a))
            .set_input(a.as_ref())
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
    pub fn neg<A>(&'b self, a: A) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        Tensor::builder()
            .set_input(a.as_ref())
            .build(self, crate::tensor::Dummy)
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
    pub fn square<A>(&'b self, a: A) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        Tensor::builder()
            .set_shape(&self.shape(a))
            .set_input(a.as_ref())
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
    pub fn reciprocal<A>(&'b self, x: A) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        Tensor::builder()
            .set_shape(&self.shape(x))
            .set_input(x.as_ref()).build(self, math_ops::Reciprocal)
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
    pub fn ceil<A>(&'b self, a: A) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        Tensor::builder()
            .set_shape(&self.shape(a))
            .set_input(a.as_ref())
            .build(self, math_ops::Ceil)
    }

    /// Returns a binary tensor.
    ///
    /// # Panics
    /// When broadcast is impossible
    pub fn greater<A, B>(&'b self, a: A, b: B) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
            B: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        Tensor::builder()
            .set_inputs(&[a.as_ref(), b.as_ref()])
            .build(self, math_ops::Greater)
    }

    /// Returns a binary tensor.
    ///
    /// # Panics
    /// When broadcast is impossible
    pub fn greater_equal<A, B>(&'b self, a: A, b: B) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
            B: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        Tensor::builder()
            .set_inputs(&[a.as_ref(), b.as_ref()])
            .build(self, math_ops::GreaterEqual)
    }

    /// Returns a binary tensor.
    ///
    /// # Panics
    /// When broadcast is impossible
    pub fn lesser<A, B>(&'b self, a: A, b: B) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
            B: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        Tensor::builder()
            .set_inputs(&[a.as_ref(), b.as_ref()])
            .build(self, math_ops::Lesser)
    }

    /// Returns a binary tensor.
    ///
    /// # Panics
    /// When broadcast is impossible
    pub fn lesser_equal<A, B>(&'b self, a: A, b: B) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
            B: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        Tensor::builder()
            .set_inputs(&[a.as_ref(), b.as_ref()])
            .build(self, math_ops::LesserEqual)
    }

    /// Elementwise logistic sigmoid function.
    pub fn sigmoid<A, B>(&'b self, x: A) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        Tensor::builder()
            .set_shape(&self.shape(x))
            .set_input(x.as_ref()).build(self, activation_ops::Sigmoid)
    }

    /// Elementwise exponential linear unit.
    ///
    /// See https://arxiv.org/abs/1511.07289
    pub fn elu<A>(&'b self, x: A, alpha: F) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        Tensor::builder()
            .set_shape(&self.shape(x))
            .set_input(x.as_ref()).build(self, activation_ops::ELU { alpha })
    }

    /// Elementwise rectified linear unit.
    pub fn relu<A>(&'b self, x: A) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        Tensor::builder()
            .set_shape(&self.shape(x))
            .set_input(x.as_ref()).build(self, activation_ops::ReLU)
    }

    /// Elementwise leaky relu.
    ///
    /// In common, `alpha` is around 0.1 ~ 0.2.
    ///
    /// See http://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf
    pub fn leaky_relu<A: 'a>(&'b self, x: A, alpha: F) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        self.maximum(&x, self.scalar(alpha) * x.as_ref())
    }

    /// Elementwise softplus.
    pub fn softplus<A>(&'b self, x: A) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        Tensor::builder()
            .set_shape(&self.shape(x))
            .set_input(x.as_ref()).build(self, activation_ops::Softplus)
    }

    /// Computes `log(sum(exp(x)))` along specified axis.
    ///
    /// `axis` can be negative.
    pub fn reduce_logsumexp<A>(&'b self, x: A, axis: isize, keep_dim: bool) -> ScopedTensor<'a, 'b, F>
    where
        A: AsRef<ScopedTensor<'a, 'b, F>>
    {
        let op = math_ops::LogSumExp {
            axis,
            keep_dims: keep_dim,
        };
        Tensor::builder().set_input(x.as_ref()).build(self, op)
    }

    /// Log softmax function.
    ///
    /// Computes `softmax(x)` along specified axis and
    /// takes logarithm of it.
    /// `axis` can be negative.
    pub fn log_softmax<A>(&'b self, x: A, axis: isize) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        Tensor::builder()
            .set_shape(&self.shape(x))
            .set_input(x.as_ref()).build(self, xent_ops::LogSoftmax { axis })
    }

    /// Computes softmax along specified axis
    ///
    /// `axis` can be negative.
    pub fn softmax<A>(&'b self, x: A, axis: isize) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        let op = activation_ops::Softmax { axis };
        Tensor::builder().set_input(x.as_ref()).build(self, op)
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
    pub fn sigmoid_cross_entropy<A, B>(&'b self, y: A, t: B) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
            B: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        let op = xent_ops::SigmoidCrossEntropy;
        Tensor::builder()
            .set_shape(&self.shape(y))
            .set_inputs(&[y.as_ref(), t.as_ref()])
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
    pub fn softmax_cross_entropy<A, B>(&'b self, y: A, t: B) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
            B: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        let op = xent_ops::SoftmaxCrossEntropy;
        Tensor::builder().set_inputs(&[y.as_ref(), t.as_ref()]).build(self, op)
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
    pub fn sparse_softmax_cross_entropy<A, B>(&'b self, y: A, t: B) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
            B: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        let op = xent_ops::SparseSoftmaxCrossEntropy;
        Tensor::builder().set_inputs(&[y.as_ref(), t.as_ref()]).build(self, op)
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
    pub fn matmul<A, B>(&'b self, a: A, b: B) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
            B: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        let op = dot_ops::MatMul {
            transpose_a: false,
            transpose_b: false,
        };
        Tensor::builder().set_inputs(&[a.as_ref(), b.as_ref()]).build(self, op)
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
    pub fn matmul_t<A, B>(
        &'b self,
        a: A,
        b: B,
        transpose_a: bool,
        transpose_b: bool,
    ) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
            B: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        let op = dot_ops::MatMul {
            transpose_a,
            transpose_b,
        };
        Tensor::builder().set_inputs(&[a.as_ref(), b.as_ref()]).build(self, op)
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
    pub fn tensordot<A, B, C, D>(
        &'b self,
        a: A,
        b: B,
        a_axes: C,
        b_axes: D,
    ) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
            B: AsRef<ScopedTensor<'a, 'b, F>>,
            C: AsRef<ScopedTensor<'a, 'b, F>>,
            D: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        // Preprocess
        let pre = &Tensor::builder()
            .set_inputs(&[a.as_ref(), b.as_ref(), a_axes.as_ref(), b_axes.as_ref()])
            .build(self, dot_ops::TensordotPreprocess);
        let final_shape = self.nth_tensor(pre, 0);
        let perm_a = self.nth_tensor(pre, 1);
        let perm_b = self.nth_tensor(pre, 2);
        let new_shape_a = self.nth_tensor(pre, 3);
        let new_shape_b = self.nth_tensor(pre, 4);

        let a_reshaped = self.reshape(self.transpose(a, perm_a), new_shape_a);
        let b_reshaped = self.reshape(self.transpose(b, perm_b), new_shape_b);

        // matmul
        let mm = self.matmul(a_reshaped, b_reshaped);
        self.reshape(mm, final_shape)
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
    pub fn batch_matmul_t<A, B>(
        &'b self,
        a: A,
        b: B,
        trans_a: bool,
        trans_b: bool,
    ) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
            B: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        let op = dot_ops::BatchMatMul {
            transpose_a: trans_a,
            transpose_b: trans_b,
        };
        Tensor::builder().set_inputs(&[a.as_ref(), b.as_ref()]).build(self, op)
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
    pub fn batch_matmul<A, B>(&'b self, a: A, b: B) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
            B: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        let op = dot_ops::BatchMatMul {
            transpose_a: false,
            transpose_b: false,
        };
        Tensor::builder().set_inputs(&[a.as_ref(), b.as_ref()]).build(self, op)
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
    pub fn setdiff1d<A, B>(&'b self, a: A, b: B) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
            B: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        let op = array_ops::SetDiff1D;
        Tensor::builder().set_inputs(&[a.as_ref(), b.as_ref()]).build(self, op)
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
    pub fn transpose<A, B>(&'b self, x: A, perm: B) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
            B: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        let op = math_ops::Transpose { invert_axes: false };
        Tensor::builder().set_inputs(&[x.as_ref(), perm.as_ref()]).build(self, op)
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
    pub fn split<A>(&'b self, x: A, sizes: &[usize], axis: isize) -> Vec<ScopedTensor<'a, 'b, F>>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        let len = sizes.len();
        let mut ret = Vec::with_capacity(len);
        for i in 0..len {
            let mut start_index = 0usize;
            for &size in sizes[..i].iter() {
                start_index += size;
            }
            let end_index = start_index + sizes[i];
            ret.push(Tensor::builder().set_input(x.as_ref()).build(
                self,
                array_ops::Split {
                    start_index: start_index as isize,
                    end_index: end_index as isize,
                    axis,
                },
            ));
        }
        ret
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
    pub fn slice<A>(&'b self, x: A, starts: &[isize], ends: &[isize]) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
    {
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
            .set_input(x.as_ref()).build(self, array_ops::Slice { indices })
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
    pub fn concat<A>(&'b self, _tensors: &[A], axis: isize) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        let op = array_ops::Concat { axis };
        let tensors = _tensors.iter().map(|t| t.as_ref()).collect::<Vec<_>>();
        Tensor::builder().set_inputs(&tensors).build(self, op)
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
    pub fn gather_common<A, B>(&'b self, param: A, indices: B, axis: isize) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
            B: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        let op = array_ops::Gather {
            axis,
            should_normalize_negative_indices: true,
        };
        Tensor::builder()
            .set_inputs(&[indices.as_ref(), param.as_ref()])
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
    pub fn gather<A, B>(&'b self, param: A, indices: B, axis: isize) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
            B: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        let op = array_ops::Gather {
            axis,
            should_normalize_negative_indices: false,
        };
        Tensor::builder()
            .set_inputs(&[indices.as_ref(), param.as_ref()])
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
    pub fn normalize<A: 'a, B>(&'b self, _x: A, _axes: B) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
            B: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        let x = _x.as_ref();
        let axes = _axes.as_ref();
        let mean = self.reduce_mean(x.as_ref(), axes, true);
        let centered = x - mean;
        let variance = self.reduce_mean(self.square(centered), axes, true);
        let em5 = self.scalar(F::from(1e-5).unwrap());
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
    pub fn batch_norm<A: 'a, B: 'a, C: 'a>(&'b self, x: A, scale: B, shift: C) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
            B: AsRef<ScopedTensor<'a, 'b, F>>,
            C: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        self.normalize(x.as_ref(), self.axes(&[0])) * scale.as_ref() + shift.as_ref()
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
    pub fn scalar(&'b self, val: F) -> ScopedTensor<'a, 'b, F> {
        let op = const_gen_ops::Scalar { val };
        Tensor::builder()
            .set_shape(&self.convert_to_tensor(crate::ndarray_ext::scalar_shape()))
            .build(self, op)
    }

    /// Outputs values sampled from the normal distribution.
    pub fn random_normal<A>(&'b self, shape: A, mean: f64, stddev: f64) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        self.random_normal_rng(Default::default(), shape, mean, stddev)
    }

    /// Outputs values sampled from the normal distribution.
    pub fn random_normal_rng<A, R: Rng + Send + 'static>(
        &'b self,
        arr_rng: ArrRng<F, R>,
        shape: A,
        mean: f64,
        stddev: f64,
    ) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        Tensor::builder()
            .set_input(shape.as_ref())
            .set_shape(shape.as_ref())
            .build(self, random_ops::RandomNormal::new(arr_rng, mean, stddev))
    }

    /// Outputs values sampled from the uniform distribution.
    pub fn random_uniform<A>(&'b self, shape: A, min: f64, max: f64) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        self.random_uniform_rng(Default::default(), shape, min, max)
    }

    /// Outputs values sampled from the uniform distribution.
    ///
    /// See https://github.com/raskr/rust-autograd/issues/1.
    pub fn random_uniform_rng<A, R: Rng + Send + 'static>(
        &'b self,
        arr_rng: ArrRng<F, R>,
        shape: A,
        min: f64,
        max: f64,
    ) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        let shape = shape;
        Tensor::builder()
            .set_input(shape.as_ref())
            .set_shape(shape.as_ref())
            .build(self, random_ops::RandomUniform::new(arr_rng, min, max))
    }

    /// Outputs values sampled from the standard normal distribution.
    pub fn standard_normal<A>(&'b self, shape: A) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        self.standard_normal_rng(Default::default(), shape)
    }

    /// Outputs values sampled from the standard normal distribution.
    ///
    /// See https://github.com/raskr/rust-autograd/issues/1.
    pub fn standard_normal_rng<A, R: Rng + Send + 'static>(
        &'b self,
        arr_rng: ArrRng<F, R>,
        shape: A,
    ) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        let shape = shape;
        Tensor::builder()
            .set_input(shape.as_ref())
            .set_shape(shape.as_ref())
            .build(self, random_ops::StandardNormal::new(arr_rng))
    }

    /// Outputs values sampled from the standard uniform distribution.
    pub fn standard_uniform<A>(&'b self, shape: A) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        self.standard_uniform_rng(Default::default(), shape)
    }

    /// Outputs values sampled from the standard uniform distribution.
    ///
    /// See https://github.com/raskr/rust-autograd/issues/1.
    pub fn standard_uniform_rng<A, R: Rng + Send + 'static>(
        &'b self,
        arr_rng: ArrRng<F, R>,
        shape: A,
    ) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        let shape = shape;
        Tensor::builder()
            .set_input(shape.as_ref())
            .set_shape(shape.as_ref())
            .build(self, random_ops::StandardUniform::new(arr_rng))
    }

    /// Outputs values sampled from the bernoulli distribution.
    pub fn bernoulli<A>(&'b self, shape: A, p: f64) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        self.bernoulli_rng(Default::default(), shape, p)
    }

    /// Outputs values sampled from the bernoulli distribution.
    ///
    /// See https://github.com/raskr/rust-autograd/issues/1.
    pub fn bernoulli_rng<A, R: Rng + Send + 'static>(
        &'b self,
        arr_rng: ArrRng<F, R>,
        shape: A,
        p: f64,
    ) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        let shape = shape;
        Tensor::builder()
            .set_input(shape.as_ref())
            .set_shape(shape.as_ref())
            .build(self, random_ops::Bernoulli::new(arr_rng, p))
    }

    /// Outputs values sampled from the exponential distribution.
    pub fn random_exp<A>(&'b self, shape: A, lambda: f64) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        self.random_exp_rng(Default::default(), shape, lambda)
    }

    /// Outputs values sampled from the exponential distribution.
    ///
    /// See https://github.com/raskr/rust-autograd/issues/1.
    pub fn random_exp_rng<A, R: Rng + Send + 'static>(
        &'b self,
        arr_rng: ArrRng<F, R>,
        shape: A,
        lambda: f64,
    ) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        let shape = shape;
        Tensor::builder()
            .set_input(shape.as_ref())
            .set_shape(shape.as_ref())
            .build(self, random_ops::Exponential::new(arr_rng, lambda))
    }

    /// Outputs values sampled from the gamma distribution.
    pub fn random_gamma<A>(&'b self, shape: A, shape_param: f64, scale: f64) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        self.random_gamma_rng(Default::default(), shape, shape_param, scale)
    }

    /// Outputs values sampled from the gamma distribution.
    ///
    /// See https://github.com/raskr/rust-autograd/issues/1.
    pub fn random_gamma_rng<A, R: Rng + Send + 'static>(
        &'b self,
        arr_rng: ArrRng<F, R>,
        shape: A,
        shape_param: f64,
        scale: f64,
    ) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        Tensor::builder()
            .set_input(shape.as_ref())
            .set_shape(shape.as_ref())
            .build(self, random_ops::Gamma::new(arr_rng, shape_param, scale))
    }

    /// Outputs values sampled from the log-normal distribution.
    pub fn log_normal<A>(&'b self, shape: A, mean: f64, stddev: f64) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        self.log_normal_rng(Default::default(), shape, mean, stddev)
    }

    /// Outputs values sampled from the log-normal distribution.
    ///
    /// See https://github.com/raskr/rust-autograd/issues/1.
    pub fn log_normal_rng<A, R: Rng + Send + 'static>(
        &'b self,
        arr_rng: ArrRng<F, R>,
        shape: A,
        mean: f64,
        stddev: f64,
    ) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        let shape = shape;
        Tensor::builder()
            .set_input(shape.as_ref())
            .set_shape(shape.as_ref())
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
    pub fn convert_to_tensor<D>(&'b self, arr: ndarray::Array<F, D>) -> ScopedTensor<'a, 'b, F>
        where
            D: ndarray::Dimension,
    {
        let arr = arr.into_dyn();
        let shape = Tensor::builder().build(
            self,
            const_gen_ops::ConvertToTensor {
                arr: crate::ndarray_ext::shape_of(&arr),
            },
        );
        Tensor::builder()
            .set_shape(shape.as_ref())
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
    pub fn zeros<A>(&'b self, shape: A) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        Tensor::builder()
            .set_input(shape.as_ref())
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
    pub fn ones<A>(&'b self, shape: A) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        Tensor::builder()
            .set_input(shape.as_ref())
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
    pub fn range(&'b self, start: F, end: F, step: F) -> ScopedTensor<'a, 'b, F> {
        Tensor::builder()
            .set_inputs(&[&self.scalar(start), &self.scalar(end), &self.scalar(step)])
            .build(self, const_gen_ops::Range)
    }

    pub fn _range<A, B, C>(&'b self, start: A, end: B, step: C) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
            B: AsRef<ScopedTensor<'a, 'b, F>>,
            C: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        Tensor::builder()
            .set_inputs(&[start.as_ref(), end.as_ref(), step.as_ref()])
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
    pub fn conv2d<A, B>(&'b self, x: A, w: B, pad: usize, stride: usize) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
            B: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        Tensor::builder().set_inputs(&[x.as_ref(), w.as_ref()]).build(
            self,
            conv_ops::conv2d::Conv2D {
                pad,
                stride,
                dilation: 1,
            },
        )
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
    pub fn dilated_conv2d<A, B>(
        &'b self,
        x: A,
        w: B,
        pad: usize,
        stride: usize,
        dilate: usize,
    ) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
            B: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        Tensor::builder().set_inputs(&[x.as_ref(), w.as_ref()]).build(
            self,
            conv_ops::conv2d::Conv2D {
                pad,
                stride,
                dilation: dilate,
            },
        )
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
    pub fn conv2d_transpose<A, B>(
        &'b self,
        x: A,
        w: B,
        pad: usize,
        stride: usize,
    ) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
            B: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        Tensor::builder().set_inputs(&[x.as_ref(), w.as_ref()]).build(
            self,
            conv_ops::conv2d_transpose::Conv2DTranspose {
                pad,
                stride,
                dilation: 1,
            },
        )
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
    pub fn dilated_conv2d_transpose<A, B>(
        &'b self,
        x: A,
        w: B,
        pad: usize,
        stride: usize,
        dilate: usize,
    ) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
            B: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        Tensor::builder().set_inputs(&[x.as_ref(), w.as_ref()]).build(
            self,
            conv_ops::conv2d_transpose::Conv2DTranspose {
                pad,
                stride,
                dilation: dilate,
            },
        )
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
    pub fn max_pool2d<A, B>(
        &'b self,
        x: A,
        pool_size: usize,
        pad: usize,
        stride: usize,
    ) -> ScopedTensor<'a, 'b, F>
        where
            A: AsRef<ScopedTensor<'a, 'b, F>>,
    {
        Tensor::builder().set_input(x.as_ref()).build(
            self,
            conv_ops::max_pool2d::MaxPool2D {
                pad,
                stride,
                size: pool_size,
            },
        )
    }
}
