//! Provides helper functions for testing.
use crate::runtime::Feed;
use crate::tensor::Tensor;
use crate::{ndarray_ext, Float};

/// Checks the validity of `gradients` with finite difference trick.
/// For this test only, `variables` must be *shared* variables.
pub fn check_theoretical_grads<'s: 't, 't, 'v, T: Float, A>(
    objective: A,
    gradients: &'t [A],
    variables: &'t [A],
    feeds: &[Feed<'v, T>],
    eps: T,
    tol: T,
) where
    A: AsRef<Tensor<'t, 's, T>> + Copy,
{
    let s = objective.as_ref().graph;
    let objective = s.reduce_sum_to_scalar(objective);
    // backprop
    let theoretical_grads = s.eval(gradients, feeds.clone());

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
            .as_ref()
            .lock_variable_array()
            .expect("This is not a variable")
            .len();
        for i in 0..v_len as isize {
            let evacuated;

            // perturbation (+)
            unsafe {
                let head_ptr: *mut T = get_head_ptr(var_node.as_ref());
                evacuated = *head_ptr.offset(i);
                *head_ptr.offset(i) = evacuated + eps;
            }

            // eval
            let obj_pos_orig = s.eval(&[objective], feeds).remove(0).unwrap();
            let obj_pos = if obj_pos_orig.is_standard_layout() {
                obj_pos_orig
            } else {
                ndarray_ext::deep_copy(&obj_pos_orig.view())
            };

            // perturbation (-)
            unsafe {
                let head_ptr: *mut T = get_head_ptr(var_node.as_ref());
                *head_ptr.offset(i) = evacuated - eps;
            }

            // eval
            let obj_neg_orig = s.eval(&[objective], feeds).remove(0).unwrap();
            let obj_neg = if obj_neg_orig.is_standard_layout() {
                obj_neg_orig
            } else {
                ndarray_ext::deep_copy(&obj_neg_orig.view())
            };

            // restore
            unsafe {
                let head_ptr: *mut T = get_head_ptr(var_node.as_ref());
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
        .lock_variable_array_mut()
        .expect("This is not a variable")
        .as_mut_ptr()
}
