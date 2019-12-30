//! Defining things related to `ag::Graph`.

use crate::{tensor::TensorInternal, Float};
use std::cell::UnsafeCell;

/// Generator of `Tensor` objects.
///
/// Use `autograd::with` function to instantiate this.
///
/// ```
/// use autograd as ag;
///
/// ag::with(|graph1: &mut ag::Graph<f32>| {
///     // Creating some nodes (tensors) in this graph.
///     let a = graph1.zeros(&[2, 3]);
///     let b = graph1.ones(&[2, 3]);
///
///     // Evaluate the tensors
///     (a + b).eval(&[]);
///
///     // Creating another scope (graph).
///     ag::with(|graph2: &mut ag::Graph<f32>| {
///         // `c` is valid only in this graph.
///         let c = graph2.zeros(&[3, 4]);
///
///         // Cross-scope access to what derived from `Graph` can't compile for now.
///
///         // graph1.zeros(&[2, 3])
///         // ^^^^^^ invalid access for `graph1`
///
///         // a + c
///         // ^ invalid access for `a` belonging to ``graph1`
///
///         // Manually removing all tensors in `graph2`.
///         graph2.clear();
///     });
///
///     // If you use `a` or `b` cleared from the graph here, panic would occur.
///     // ...
/// });
/// // Scope ends; all the nodes are destructed here.
/// ```
pub struct Graph<F: Float> {
    node_set: UnsafeCell<Vec<TensorInternal<F>>>,
}

impl<'a, 'b, F: Float> Graph<F> {
    pub(crate) fn install(&'b self, mut node: TensorInternal<F>) -> &'a TensorInternal<F> {
        unsafe {
            let inner = &mut *self.node_set.get();
            let id = inner.len();
            node.id = id;
            inner.push(node);
            inner.get_unchecked(id)
        }
    }

    pub(crate) fn access_node(&self, i: usize) -> &'a TensorInternal<F> {
        unsafe {
            let inner = &*self.node_set.get();
            inner
                .get(i)
                .expect("Accessing cleared node. (remove `Graph::clear` line)")
        }
    }

    /// Removes all tensors (nodes) in this graph.
    ///
    /// Be careful not to remove tensors that will be needed later.
    pub fn clear(&mut self) {
        unsafe {
            (&mut *self.node_set.get()).clear();
        }
    }
}

/// Creates a scope for a computation graph.
///
/// This is the only way to access `Graph` objects.
pub fn with<F, FN>(f: FN)
where
    F: Float,
    FN: FnOnce(&mut Graph<F>) -> () + Send,
{
    let mut g = Graph {
        node_set: UnsafeCell::new(Vec::with_capacity(128)),
    };
    f(&mut g);
}
