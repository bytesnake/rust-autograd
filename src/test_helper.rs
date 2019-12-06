use crate::tensor::Tensor;
use crate::{ndarray_ext, Scope};
use crate::{Feed, Float};
use std::cmp::Ordering;
use std::collections::btree_set::BTreeSet;

/// Checks the validity of `gradients` with finite difference trick.
/// For this test only, `variables` must be *shared* variables.
pub fn check_theoretical_grads<'k, 'v, T: Float>(
    objective: &'k Tensor<T>,
    gradients: &'k [&'k Tensor<T>],
    variables: &[&'k Tensor<T>],
    feeds: &'v [Feed<'k, 'v, T>],
    eps: T,
    tol: T,
    c: &mut Scope<T>,
) {
    let objective = c.reduce_sum_to_scalar(objective);
    // backprop
    let theoretical_grads = crate::runtime::eval(gradients, feeds.clone());

    // for each variable nodes
    for (var_node, th_grad) in variables.into_iter().zip(theoretical_grads) {
        let th_copied = if th_grad.as_ref().unwrap().is_standard_layout() {
            None
        } else {
            Some(ndarray_ext::deep_copy(&th_grad.as_ref().unwrap().view()))
        };
        let th_ptr = if let Some(ref inner) = th_copied {
            inner.as_ptr()
        } else {
            th_grad.as_ref().unwrap().as_ptr()
        };

        // for each values
        let v_len = var_node
            .get_variable_array()
            .expect("This is not a variable")
            .len();
        for i in 0..v_len as isize {
            let evacuated;

            // perturbation (+)
            unsafe {
                let head_ptr: *mut T = get_head_ptr(var_node);
                evacuated = *head_ptr.offset(i);
                *head_ptr.offset(i) = evacuated + eps;
            }

            // eval
            let obj_pos_orig = objective.eval(feeds).unwrap();
            let obj_pos = if obj_pos_orig.is_standard_layout() {
                obj_pos_orig
            } else {
                ndarray_ext::deep_copy(&obj_pos_orig.view())
            };

            // perturbation (-)
            unsafe {
                let head_ptr: *mut T = get_head_ptr(var_node);
                *head_ptr.offset(i) = evacuated - eps;
            }

            // eval
            let obj_neg_orig = objective.eval(feeds).unwrap();
            let obj_neg = if obj_neg_orig.is_standard_layout() {
                obj_neg_orig
            } else {
                ndarray_ext::deep_copy(&obj_neg_orig.view())
            };

            // restore
            unsafe {
                let head_ptr: *mut T = get_head_ptr(var_node);
                *head_ptr.offset(i) = evacuated;
            }

            let two = T::one() + T::one();
            let g_num = (obj_pos - obj_neg).scalar_sum() / (two * eps);
            let g_th = unsafe { *th_ptr.offset(i) };

            // compare
            let diff = (g_num - g_th).abs();
            if diff > tol {
                panic!(
                    "Gradient checking failed with too large error: numerical={}, theoretical={}",
                    g_num, g_th
                );
            }
        }
    }
}

fn get_head_ptr<'a, T: Float>(var_node: &Tensor<T>) -> *mut T {
    var_node
        .get_variable_array_mut()
        .expect("This is not a variable")
        .as_mut_ptr()
}

/// Traverse a graph from endpoint "t".
pub fn visit_once<'a, F, T: Float>(t: &Tensor<T>, f: &mut F)
where
    F: FnMut(&Tensor<T>) -> (),
{
    visit_once_internal(t, f, &mut BTreeSet::new())
}

fn visit_once_internal<'a, F, T: Float>(
    t: &Tensor<T>,
    f: &mut F,
    visited: &mut BTreeSet<&Tensor<T>>,
) where
    F: FnMut(&Tensor<T>) -> (),
{
    if visited.contains(&t) {
        return; // exit early
    } else {
        visited.insert(t); // first visit
    }

    f(&t);

    for child in t.inputs.iter() {
        visit_once_internal(child, f, visited)
    }
}

impl<'a, T: Float> Ord for &Tensor<T> {
    #[inline]
    /// Compares the addresses of the two tensors.
    /// This can be used for ordering-based data structures (e.g. BinaryTree).
    fn cmp(&self, other: &&Tensor<T>) -> Ordering {
        let a = (*self) as *const Tensor<T>;
        let b = (*other) as *const Tensor<T>;
        a.cmp(&b)
    }
}

impl<'a, T: Float> PartialOrd for &Tensor<T> {
    #[inline]
    /// Compares the addresses of the two tensors.
    /// This can be used for ordering-based data structures (e.g. BinaryTree).
    fn partial_cmp(&self, other: &&Tensor<T>) -> Option<Ordering> {
        Some(self.cmp(&other))
    }
}
