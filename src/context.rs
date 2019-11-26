use crate::{tensor, Tensor, Float};

pub struct Context<'a, F: Float> {
    node_set: Vec<Tensor<'a, F>>
}

impl<'a, F: Float> Context<'a, F> {
    pub(crate) fn install(&'a mut self, mut node: Tensor<'a, F>) -> &'a Tensor<'a, F> {
        let id = self.node_set.len();
        node.id = id;
        self.node_set.push(node);
        &self.node_set[id]
    }
}

pub fn scope<'a, FN, F>(mut f: FN)
where
    F: Float,
    FN: FnMut(Context<'a, F>) -> ()
{
    f(Context {
        node_set: Vec::with_capacity(128)
    });
}
