use crate::tensor::{Tensor, ScopedTensor};
use crate::Float;
use crate::FxHashMap;
use crate::Scope;
use std::cmp::Ordering;
use std::collections::binary_heap::BinaryHeap;
use std::mem;
use crate::op::Op;
use std::cell::UnsafeCell;

struct GradInfo<'a, 'b: 'a, T: Float + 'a> {
    has_gradient: bool,
    grad_called: bool,
    computed_grads: UnsafeCell<Vec<ScopedTensor<'a, 'b, T>>>,
    default_grad: Option<&'a Tensor<T>>,
}

impl<'a, 'b: 'a, T: Float> GradInfo<'a, 'b, T> {
    #[inline]
    fn new(has_gradient: bool, default_grad: Option<&'a Tensor<T>>) -> GradInfo<'a, 'b, T> {
        GradInfo {
            has_gradient,
            computed_grads: UnsafeCell::new(Vec::new()),
            grad_called: false,
            default_grad,
        }
    }
}

#[inline]
fn has_marked_child<'a, 'b: 'b, T: Float>(
    s: &'b Scope<T>,
    parent: &Tensor<T>,
    path: &FxHashMap<&'a Tensor<T>, GradInfo<'a, 'b, T>>,
) -> bool {
    let mut it = parent.get_backprop_inputs().iter();
    while let Some(child) = it.next() {
        if path.get(child.get(s)).unwrap().has_gradient {
            return true;
        }
    }
    false
}

#[inline]
fn is_wrt<'a, T: Float>(node: &Tensor<T>, wrt: &[&Tensor<T>]) -> bool {
    wrt.contains(&node)
}

// Marks `has_gradient` if each node is on the gradient propagation path.
//
// Strategy
//   1. Record all nodes that are reachable from `ys` into `ret`.
//   2. Mark the path between `ys` and `xs` as `has_gradient`.
fn make_between_nodes<'a, 'b: 'a, T: Float>(
    s: &'b Scope<T>,
    ys: &[&'a Tensor<T>],
    wrt: &[&'a Tensor<T>],
) -> FxHashMap<&'a Tensor<T>, GradInfo<'a, 'b, T>> {
    // Randomly accessible by use of each node's lookup key.
    let mut ret = FxHashMap::<&Tensor<T>, GradInfo<T>>::default();

    // Builds GradInfo while performing depth-first-search.
    // `has_gradient` properties are filled at the same time.

    // dfs_stack: (node, should_visit)
    let mut dfs_stack: Vec<(&Tensor<T>, bool)> = ys.iter().map(|&y| (y, false)).collect();
    while let Some((node, should_visit)) = dfs_stack.pop() {
        if should_visit {
            let marker =
                node.is_differentiable && (is_wrt(node, wrt) || has_marked_child(s, node, &ret));
            ret.insert(node, GradInfo::new(marker, None));
        } else {
            // Put self on the stack top (should visit next time)
            dfs_stack.push((node, true));
            // Push children as necessary
            for child in node.get_backprop_inputs() {
                let child = child.get(s);
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

pub struct GradientContext<'a, 'b: 'a, T: Float> {
    gy: ScopedTensor<'a, 'b, T>,
    xs: Vec<ScopedTensor<'a, 'b, T>>,
    y: ScopedTensor<'a, 'b, T>,
    s: &'b crate::scope::Scope<T>,
    gxs: Option<Vec<Option<ScopedTensor<'a, 'b, T>>>>,
}

impl<'a, 'b: 'a, T: Float> GradientContext<'a, 'b, T> {
    #[inline(always)]
    pub fn output_grad(&self) -> ScopedTensor<'a, 'b, T> {
        self.gy
    }

    #[inline(always)]
    pub fn output(&self) -> ScopedTensor<'a, 'b, T> {
        self.y
    }

    #[inline(always)]
    pub fn input(&self, i: usize) -> ScopedTensor<'a, 'b, T> {
        self.xs[i]
    }

    #[inline(always)]
    pub fn num_inputs(&self) -> usize {
        self.xs.len()
    }

    #[inline(always)]
    pub fn scope(&self) -> &'b crate::scope::Scope<T> {
        self.s
    }

    #[inline]
    pub fn set_input_grads(&mut self, gxs: Vec<Option<ScopedTensor<'a, 'b, T>>>) {
        self.gxs = Some(gxs);
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
pub fn symbolic_gradients<'a, 'b: 'a, T: Float>(
    ys: &[&'a Tensor<T>],
    wrt: &[&'a Tensor<T>],
    gys: &[&'a Tensor<T>],
    s: &'b Scope<T>,
) -> Vec<ScopedTensor<'a, 'b, T>> {
    assert_eq!(ys.len(), gys.len(), "`ys.len()` must match `gys.len()`");

    // Setup gradient path.
    let mut between_nodes = UnsafeCell::new(make_between_nodes(s, ys, wrt));

    unsafe {
    // Set default grads.
    for (y, gy) in ys.iter().zip(gys) {
        (&mut *between_nodes.get()).get_mut(y).unwrap().default_grad = Some(gy);
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
            // この info が、loop が 100回回ったら 100 こ、gx として外に出てる。
            let info: &mut GradInfo<_> = (&mut *between_nodes.get()).get_mut(y.inner).unwrap();
            let gy = if let Some(def) = info.default_grad {
                def
            } else {
                let gys = info.computed_grads.get();
                accumulate_grads_if_needed(gys, s);
                &(&*gys)[0]
            };
            // Call Op::grad
            let xs = y.inner.get_scoped_input(s);
            let xs_len = xs.len();
            let mut ctx = GradientContext {
                gy: s.scope(gy), xs, y: s.scope(y.inner), s, gxs: None
            };
            y.inner.op.grad(&mut ctx);
            let gxs = ctx.gxs.expect("Bad Op impl: GradientContext::set_input_grads was not called");
            debug_assert_eq!(xs_len, gxs.len());
            gxs
        };
        // Register computed gradients
        let xs = y.inner.get_backprop_inputs();
        for (gx, x) in gxs.into_iter().zip(xs) {
            let x = x.get(s);
            let mut x_info = (&mut *between_nodes.get()).get_mut(&x).unwrap();
            if x_info.has_gradient {
                if let Some(gx) = gx {
                    (&mut *x_info.computed_grads.get()).push(gx);
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
    let mut ret = Vec::with_capacity(wrt.len());
    for x in wrt {
        let msg1: &str = "Not differentiable with given tensor(s).";
        let info = (&mut *between_nodes.get()).get_mut(x).expect(msg1);
        if !info.has_gradient {
            panic!(msg1);
        }
        assert!(
            info.default_grad.is_none(),
            "Can't differentiate with objective itself"
        );
        let gxs = info.computed_grads.get();
        accumulate_grads_if_needed(gxs, s);
        ret.push(ScopedTensor{ scope: s, inner: (&mut *gxs).remove(0).inner });
    }
    ret
    }
}

struct TensorWrapper<'a, T: Float + 'a> {
    inner: &'a Tensor<T>,
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

impl<'a, T: Float> Tensor<T> {
    #[inline]
    fn wrapped(&'a self) -> TensorWrapper<'a, T> {
        TensorWrapper { inner: self }
    }
}

#[inline]
fn accumulate_grads_if_needed<'a, 'b: 'a, T: Float>(grads: *mut Vec<ScopedTensor<'a, 'b, T>>, c: &'b Scope<T>) {
    // TODO
//    if grads.len() > 1 {
//        let mut acc = c.add_n(grads.as_slice());
//        mem::swap(&mut acc, &mut grads[0]);
//        grads.truncate(1)
//    }
}
