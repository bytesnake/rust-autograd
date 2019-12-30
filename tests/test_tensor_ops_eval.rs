extern crate autograd as ag;
extern crate ndarray;
use ag::{tensor::Constant, tensor::Variable, with};
use ndarray::array;

#[test]
fn reduce_prod() {
    with(|g| {
        let v = g.constant(ag::ndarray_ext::standard_normal::<f32>(&[3, 2]));
        let z = g.reduce_prod(v, &[0, 1], false); // keep_dims=false
        let empty_shape: &[usize] = &[];
        assert_eq!(z.eval(&[]).unwrap().shape(), empty_shape);
    });
}

#[test]
fn argmax() {
    with(|g| {
        let x = g.constant(array![[3., 4.], [5., 6.]]);
        let y = g.argmax(x, -1, false);
        assert_eq!(y.eval(&[]), Some(ndarray::arr1(&[1., 1.]).into_dyn()));
    });
}

#[test]
fn argmax_with_multi_max_args() {
    with(|g| {
        let x = g.constant(array![1., 2., 3., 3.]);
        let y = g.argmax(x, 0, false);
        assert_eq!(2., y.eval(&[]).unwrap()[ndarray::IxDyn(&[])]);
    });
}

#[test]
fn reduce_mean() {
    with(|g| {
        let v = g.variable(array![2., 3., 4.]);
        let z = g.reduce_mean(v, &[0], false); // keep_dims=false
        assert_eq!(3., z.eval(&[]).unwrap()[ndarray::IxDyn(&[])]);
    });
}

#[test]
fn reduce_grad() {
    with(|g| {
        let v = g.variable(array![2., 3., 4.]);
        let z = g.reduce_mean(v, &[0], false); // keep_dims=false
        let g = g.grad(&[z], &[v])[0];
        assert_eq!(g.eval(&[]).unwrap().shape(), &[3]);
    });
}

#[test]
fn transpose_matmul() {
    with(|g| {
        let x = g.constant(array![[0., 1.], [2., 3.]]);
        let w = g.constant(array![[0., 1.], [2., 3.]]);
        let w2 = g.transpose(w, &[1, 0]);
        let mm = g.matmul(x, w2);
        assert_eq!(
            mm.eval(&[]).unwrap().as_slice().unwrap(),
            &[1., 3., 3., 13.]
        );
    });
}
