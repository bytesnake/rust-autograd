extern crate autograd as ag;
extern crate ndarray;

struct MultiOutputOp;

impl ag::op::Op<f32> for MultiOutputOp {
    fn name(&self) -> &str {
        "MultiOutputOp"
    }

    fn compute(&self, ctx: &mut ag::op::OpComputeContext<f32>) {
        let a = ag::ndarray_ext::zeros(&[2, 3]);
        let b = ag::ndarray_ext::zeros(&[1, 3]);
        ctx.push_output(Ok(ag::ArrRepr::Owned(a)));
        ctx.push_output(Ok(ag::ArrRepr::Owned(b)));
    }

    fn grad(&self, ctx: &mut ag::op::OpGradientContext<f32>) {
        ctx.set_input_grads(vec![None; 2])
    }
}

#[test]
fn test_nth_tensor() {
    ag::with(|g| {
        let a = ag::Tensor::builder().build(g, MultiOutputOp);
        let b = g.nth_tensor(a, 1);
        let c = g.exp(b);
        g.eval(&[c], &[]);
    });
}

#[test]
fn test_hook() {
    ag::with(|g| {
        let a: ag::Tensor<f32> = g.ones(&[4, 2]).show();
        let b: ag::Tensor<f32> = g.zeros(&[2, 3]).show_shape();
        let c = g.matmul(a, b).print_any("aaa");
        g.eval(&[c], &[]);
    });
    ag::with(|g: &mut ag::Graph<_>| {
        let x = g.placeholder(&[]);
        let y = g.placeholder(&[]);
        let z = 2. * x * x + 3. * y + 1.;

        // dz/dy
        let gy = &g.grad(&[z], &[y])[0];
        println!("{:?}", gy.eval(&[])); // => Some(3.)

        // dz/dx (requires to fill the placeholder `x`)
        let gx = &g.grad(&[z], &[x])[0];
        println!("{:?}", gx.eval(&[x.given(ag::ndarray::arr0(2.).view())])); // => Some(8.)

        // ddz/dx (differentiates `z` again)
        let ggx = &g.grad(&[gx], &[x])[0];
        println!("{:?}", ggx.eval(&[])); // => Some(4.)
    });
}
