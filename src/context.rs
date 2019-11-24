use crate::{Tensor, Float};
use crate::tensor::TensorCore;

pub struct Context<F: Float> {
    node_set: Vec<Tensor<F>>
}

impl<F: Float> Context<F> {
    pub(crate) fn install(&mut self, mut node: Tensor<F>) -> &Tensor<F> {
        let id = self.node_set.len();
        {
            let mut guard = node.id.write().unwrap();
            *guard = id;
        }
        self.node_set.push(node);
        &self.node_set[id]
    }
}

pub fn scope<FN, F>(mut f: FN)
where
    F: Float,
    FN: FnMut(Context<F>) -> ()
{
    f(Context {
        node_set: Vec::with_capacity(128)
    });
}

impl crate::context::Context<f32> {
    fn foo(&mut self) {
        let t1 = self.install(Tensor::dummy());
        let t2 = self.install(Tensor::dummy());
    }
}

//fn foo() {
//    scope(|ctx| {
//    })
//}
