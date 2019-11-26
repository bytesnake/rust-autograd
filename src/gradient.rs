use crate::tensor::Tensor;
use crate::Context;
use crate::Float;
use crate::FxHashMap;
use std::cmp::Ordering;
use std::collections::binary_heap::BinaryHeap;
use std::mem;
use std::sync::Arc;

struct GradInfo<'a, T: Float + 'a> {
    has_gradient: bool,
    grad_called: bool,
    computed_grads: Vec<&'a Tensor<'a, T>>,
    default_grad: Option<&'a Tensor<'a, T>>,
}

impl<'a, T: Float> GradInfo<'a, T> {
    #[inline]
    fn new(has_gradient: bool, default_grad: Option<&'a Tensor<'a, T>>) -> GradInfo<'a, T> {
        GradInfo {
            has_gradient,
            computed_grads: Vec::new(),
            grad_called: false,
            default_grad,
        }
    }
}

#[inline]
fn has_marked_child<'a, T: Float>(
    parent: &Tensor<'a, T>,
    path: &FxHashMap<&'a Tensor<'a, T>, GradInfo<T>>,
) -> bool {
    let mut it = parent.get_backprop_inputs().iter();
    while let Some(child) = it.next() {
        if path.get(child.val).unwrap().has_gradient {
            return true;
        }
    }
    false
}

#[inline]
fn is_wrt<'a, T: Float>(node: &Tensor<'a, T>, wrt: &[&Tensor<'a, T>]) -> bool {
    wrt.contains(&node)
}

// Marks `has_gradient` if each node is on the gradient propagation path.
//
// Strategy
//   1. Record all nodes that are reachable from `ys` into `ret`.
//   2. Mark the path between `ys` and `xs` as `has_gradient`.
fn make_between_nodes<'a, T: Float>(
    ys: &[&'a Tensor<'a, T>],
    wrt: &[&'a Tensor<'a, T>],
) -> FxHashMap<&'a Tensor<'a, T>, GradInfo<'a, T>> {
    // Randomly accessible by use of each node's lookup key.
    let mut ret = FxHashMap::<&Tensor<'a, T>, GradInfo<'a, T>>::default();

    // Builds GradInfo while performing depth-first-search.
    // `has_gradient` properties are filled at the same time.

    // dfs_stack: (node, should_visit)
    let mut dfs_stack: Vec<(&Tensor<'a, T>, bool)> = ys.iter().map(|&y| (y, false)).collect();
    while let Some((node, should_visit)) = dfs_stack.pop() {
        if should_visit {
            let marker =
                node.is_differentiable && (is_wrt(node, wrt) || has_marked_child(node, &ret));
            ret.insert(node, GradInfo::new(marker, None));
        } else {
            // Put self on the stack top (should visit next time)
            dfs_stack.push((node, true));
            // Push children as necessary
            for child in node.get_backprop_inputs() {
                let child = &child.val;
                if ret.get(node).is_none() {
                    if child.is_source() || !child.is_differentiable {
                        // Add to result, but don't allow any more recursive search
                        // because there will be no `wrt` nodes in this direction....
                        ret.insert(
                            child,
                            GradInfo::new(child.is_differentiable && is_wrt(child, wrt), None),
                        );
                    } else {
                        // Recurse
                        dfs_stack.push((child, false));
                    }
                }
            }
        }
    }
    ret
}

#[test]
fn test_gradient_path() {
    // dummy graph
    // y = 3 * x1 * x1 + 5 * x2 + x3;
    let x1: Tensor<f64> = crate::ops::placeholder(&[]);
    let x2 = crate::ops::placeholder(&[]);
    let x3 = crate::ops::placeholder(&[]);
    let a = 3. * x1; // rank 1
    let b = a * x1; // rank 2
    let c = 5. * x2; // rank 1
    let d = b + c; // rank 3
    let y = d + x3; // rank 4
    let path = make_between_nodes(&[y], &[x1, x2]);

    assert!(path.contains_key(x1));
    assert!(path.contains_key(x2));
    assert!(path.contains_key(x3));
    assert!(path.contains_key(a));
    assert!(path.contains_key(b));
    assert!(path.contains_key(c));
    assert!(path.contains_key(d));
    assert!(path.contains_key(y));
    assert_eq!(path.len(), 10); // number of nodes in the grad path

    // Connection test
    for node in [x1, x2, a, c, b, d, y].iter() {
        if !path.get(node).unwrap().has_gradient {
            panic!("{} is not has_gradient", node.op.name());
        }
    }
    if path.get(x3).unwrap().has_gradient {
        panic!("{} should not be has_gradient", x3.op.name());
    }
}

/// Returns symbolic gradient tensors of `xs`.
///
/// This computes partial derivatives of `ys` with `xs` and returns the
/// gradients. This is achieved by building a subgraph between `ys` and
/// `xs` in reverse order from user's graph definition.
/// `gys` are already known gradients of `ys`'s outputs.
///
/// NOTE: Nodes that do not have gradients won't be included in the subgraph to avoid
/// unnecessary computation.
pub fn symbolic_gradients<'a, T: Float>(
    ys: &[&'a Tensor<'a, T>],
    wrt: &[&'a Tensor<'a, T>],
    gys: &[&'a Tensor<'a, T>],
    c: &mut Context<'a, T>
) -> Vec<&'a Tensor<'a, T>> {
    assert_eq!(ys.len(), gys.len(), "`ys.len()` must match `gys.len()`");

    // Setup gradient path.
    let mut path = make_between_nodes(ys, wrt);

    // Set default grads.
    for (y, gy) in ys.iter().zip(gys) {
        path.get_mut(y).unwrap().default_grad = Some(gy);
    }

    // Prepare a heap with given ys.
    let mut heap = ys
        .into_iter()
        .map(|y| y.wrapped())
        .collect::<BinaryHeap<TensorWrapper<T>>>();

    // Backprop.
    // Starts with `ys`.
    while let Some(y) = heap.pop() {
        let gxs = {
            let info = &mut path.get_mut(y.inner).unwrap();
            let gy = if let Some(def) = info.default_grad {
                def
            } else {
                let gys = &mut info.computed_grads;
                accumulate_grads_if_needed(gys, c);
                &gys[0]
            };
            // Call Op::grad
            let xs = y.inner.get_input_refs();
            let gxs = y.inner.op.grad(gy, xs.as_slice(), y.inner, c);
            debug_assert_eq!(xs.len(), gxs.len());
            gxs
        };
        // Register computed gradients
        let xs = y.inner.get_backprop_inputs();
        for (gx, x) in gxs.into_iter().zip(xs) {
            let mut x_info = &mut path.get_mut(&x.val).unwrap();
            if x_info.has_gradient {
                if let Some(gx) = gx {
                    x_info.computed_grads.push(gx);
                    // update heap
                    if !x.is_source() && !x_info.grad_called {
                        x_info.grad_called = true;
                        heap.push(x.wrapped());
                    }
                }
            }
        }
    }

    // Aggregate and return xs's gradients
    wrt.iter()
        .map(|x| {
            let msg1: &str = "Not differentiable with given tensor(s).";
            let info = path.get_mut(x).expect(msg1);
            if !info.has_gradient {
                panic!(msg1);
            }
            assert!(
                info.default_grad.is_none(),
                "Can't differentiate with objective itself"
            );
            let gxs = &mut info.computed_grads;
            accumulate_grads_if_needed(gxs, c);
            gxs.remove(0)
        })
        .collect::<Vec<&Tensor<'a, T>>>()
}

struct TensorWrapper<'a, T: Float + 'a> {
    inner: &'a Tensor<'a, T>,
}

impl<'a, T: Float> Ord for TensorWrapper<'a, T> {
    // Compares the ranks in topological ordering
    fn cmp(&self, other: &Self) -> Ordering {
        self.inner.top_rank.cmp(&other.inner.top_rank)
    }
}

impl<'a, T: Float> PartialOrd for TensorWrapper<'a, T> {
    #[inline]
    // Compares the ranks in topological ordering
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.inner.top_rank.cmp(&other.inner.top_rank))
    }
}

impl<'a, T: Float> Eq for TensorWrapper<'a, T> {}

impl<'a, T: Float> PartialEq for TensorWrapper<'a, T> {
    #[inline]
    fn eq(&self, other: &TensorWrapper<'a, T>) -> bool {
        self.inner.id() == other.inner.id()
    }
}

impl<'a, T: Float> Tensor<'a, T> {
    #[inline]
    fn wrapped(&'a self) -> TensorWrapper<'a, T> {
        TensorWrapper { inner: self }
    }
}

#[inline]
fn accumulate_grads_if_needed<'a, T: Float>(grads: &mut Vec<&'a Tensor<'a, T>>, c: &mut Context<'a, T>) {
    if grads.len() > 1 {
        let mut acc = c.add_n(grads.as_slice());
        mem::swap(&mut acc, &mut grads[0]);
        grads.truncate(1)
    }
}
