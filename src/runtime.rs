use crate::ndarray_ext::{NdArray, NdArrayView, NdArrayViewMut};
use crate::op;
use crate::tensor::Tensor;
use crate::Float;
use crate::FxHashMap;
//use crate::arrayvec::ArrayVec;
use ndarray;
use std::mem;
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard, TryLockResult};

/// Helper structure for batched evaluation.
///
/// Use this in case `ag::eval` doesn't help.
///
/// ```
/// extern crate autograd as ag;
/// extern crate ndarray as nd;
///
/// let ref a = ag::placeholder(&[]);
/// let ref x = a + a;
/// let ref y = a * a;
/// let ref z = a / a;
///
/// ag::Eval::new()
///     .push(&y)
///     .extend(&[y, z])
///     .run(&[ag::Feed(a, nd::arr0(2.).into_dyn().view())]);  // Do eval
/// ```
pub struct Eval<'k, T: Float> {
    buf: Vec<&'k Tensor<T>>,
}

impl<'c, 'k, 'v, T: Float> Eval<'k, T> {
    #[inline]
    /// Instantiates a new evaluation session.
    pub fn new() -> Self {
        Eval { buf: Vec::new() }
    }

    #[inline]
    /// Appends a tensor to the back of the evaluation targets.
    pub fn push(&mut self, x: &'k Tensor<T>) -> &mut Self {
        self.buf.push(x);
        self
    }

    #[inline]
    /// Extends the evaluation targets with `xs`.
    pub fn extend<A>(&mut self, xs: &'k [A]) -> &mut Self
    where
        A: AsRef<Tensor<T>>,
    {
        self.buf.extend(xs.iter().map(|x| x.as_ref()));
        self
    }

    #[inline]
    /// Evaluates the buffered tensors.
    ///
    /// `feeds` is a stream of `(placeholder tensor, its value)`
    pub fn run(&'k self, feeds: &'c [crate::runtime::Feed<'k, 'v, T>]) -> Vec<Option<NdArray<T>>> {
        eval(&self.buf, feeds)
    }
}

/// Context in evaluation of `node`.
// When instantiate this structure, be careful that the internal pointers are safely dereferenced.
pub struct OpComputeContext<'value, 'lock, T: Float> {
    node: *const Tensor<T>,
    xs: Vec<Input<'value, T>>,
    // these locks live until this context ends.
    read_lock_register: Vec<Option<RwLockReadGuard<'lock, NdArray<T>>>>,
    write_lock_register: Vec<Option<RwLockWriteGuard<'lock, NdArray<T>>>>
}

impl<'v, 'a, T: Float> OpComputeContext<'v, 'a, T> {

    #[inline]
    fn new(node: &Tensor<T>, xs: Vec<Input<'v, T>>) -> Self {
        OpComputeContext {
            node: node as *const _,
            read_lock_register: crate::none_vec(xs.len()),
            write_lock_register: crate::none_vec(xs.len()),
            xs,
        }
    }

    pub fn input(&mut self, i: usize) -> NdArrayView<T> {
        match self.xs[i] {
            Input::Free(ref a) => {
                a.view()
            }
            Input::RwLock(ref lock) => unsafe {  // xs[i] is a variable tensor
                if self.write_lock_register[i].is_some() {
                    OpComputeContext::deadlock((&*self.node), i);
                }
                // lock.
                self.read_lock_register[i] = Some((&**lock).read().unwrap());  // mut
                if let Some(ref a) = self.read_lock_register[i] {
                    return a.view();
                }
                unreachable!();
            }
        }
    }

    pub fn input_mut(&mut self, i: usize) -> NdArrayViewMut<T> {
        unsafe {
            match self.xs[i] {
                Input::RwLock(ref lock) => {
                    if self.read_lock_register[i].is_some() {
                        OpComputeContext::deadlock((&*self.node), i);
                    }
                    if self.write_lock_register[i].is_some() {
                        OpComputeContext::deadlock((&*self.node), i);
                    }
                    // lock.
                    self.write_lock_register[i] = Some((&**lock).write().unwrap());
                    if let Some(ref mut a) = self.write_lock_register[i] {
                        return a.view_mut();
                    }
                    unreachable!();
                },
                _ => panic!("`{}` is not a variable tensor", (&*self.node).op.name())
            }
        }
    }

    fn deadlock(node: &Tensor<T>, i: usize) {
        panic!("{} required access for its {}th variable input, which is locked by self.",
               node.op.name(), i);
    }

    #[inline]
    pub fn input_raw(&self, idx: usize) -> &Input<'v, T> {
        &self.xs[idx]
    }

//    pub enum ImmutableArray<'a, 'b, T: Float> {
//      Free(NdArrayView<'a, T>),
//      WouldBlock(RwLockReadGuard<'b, NdArray<T>>),
//    }

//
//    /// It is recommended that the caller cache the return value of this.
//    /// caller が mut な x か immut な x を求めてくるかわからないので、それは caller に委ねるという方針。
//    /// しかし、read したものではなく、RwLock の参照を与えてしまうのも1つ。
//    pub fn find_input(&self, idx: usize) -> ImmutableArray<'s, T> {
//        unsafe {
//            let in_node: &Tensor<T> = &(&*self.node).inputs[idx];
//
//            // This is placeholder
//            if in_node.is_placeholder {
//                let feed_store = &*self.feed_store;
//                // f より長生きする 's が要求されているが、実際は f になってるぞ。
//                let ret: Option<&NdArrayView<T>> = feed_store.get(&(in_node as *const _));
//                return ImmutableArray::Free(feed_store.get(&(in_node as *const _)).unwrap().clone())
//            }
//
//            // Use variable array as immutable
//            if let Some(ro_guard) = in_node.get_variable_array() {
//                return ImmutableArray::WouldBlock(ro_guard)
//            }
//
//            // Use constant array
//            if let Some(ref arr) = in_node.get_constant_array() {
//                return ImmutableArray::Free(arr.view())
//            }
//
//            // Search output of other nodes
//            let info: &ValueInfo<T> = &self.info_list[idx];
//            match info.ty {
//                ValueType::Owned => {
//                    let storage = &*self.owned_storage;
//                    ImmutableArray::Free(storage[info.key].view())
//                }
//                ValueType::View => {
//                    // Clone the view
//                    let storage = &*self.view_storage;
//                    ImmutableArray::Free(storage[info.key].clone())
//                }
//                _ => {
//                    unreachable!()
//                }
//            }
//        }
//    }
}

//pub fn eval_new<'slice, 'node, 'feed, K, T>(
//    tensors: &'node [K],
//    feeds: &'slice [Feed<'node, 'feed, T>],
//) -> Vec<Option<NdArray<T>>>
//    where
//        K: AsRef<Tensor<T>>,
//        T: Float,
//{
//
//}


#[derive(Clone, Copy, PartialEq)]
enum ValueType {
    Owned,
    View,
    NoOutput,
}

#[derive(Clone)]
struct ValueInfo<T: Float> {
    ty: ValueType,
    // key to lookup the value
    key: usize,
    // Owned array
    value: Option<NdArray<T>>,
}

impl<T: Float> ValueInfo<T> {
    #[inline]
    fn new(ty: ValueType, key: usize) -> ValueInfo<T> {
        ValueInfo {
            ty,
            key,
            value: None,
        }
    }
}

struct NodeMetadata<T: Float> {
    info_list: Vec<ValueInfo<T>>, // ys
    contains_no_output: bool,
}

pub enum Input<'a, T: Float> {
    Free(NdArrayView<'a, T>),
    RwLock(*const RwLock<NdArray<T>>)
}

/// An object sent to `ag::Eval`, `ag::eval` or `Tensor::eval` to fill a placeholder tensor
///
/// The first argument should be a tensor made from `ag::placeholder`.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let x = ag::placeholder(&[2]);
///
/// // Fills placeholder, then eval
/// let arr = ndarray::arr1(&[1., 1.]).into_dyn();
/// x.eval(&[ag::Feed(&x, arr.view())]);
/// ```
pub struct Feed<'k, 'f, T: Float>(
    pub &'k Tensor<T>,                             // a placeholder tensor
    pub ndarray::ArrayView<'f, T, ndarray::IxDyn>, // its value
);

/// Evaluates given symbolic tensors.
///
/// Each return value can be `None`;
/// for example, evaluation of `gradient_descent_ops::*`
/// would result in `None`.
///
/// NOTE: All the runtime errors are not reported by return values, but by "panic"
/// for convenience.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let ref a = ag::zeros(&[2]);
/// let ref b = ag::ones(&[2]);
///
/// // eval two tensors at once.
/// let evaluated = ag::eval(&[a, b], &[]);
/// assert_eq!(evaluated[0], Some(ndarray::arr1(&[0., 0.]).into_dyn()));
/// assert_eq!(evaluated[1], Some(ndarray::arr1(&[1., 1.]).into_dyn()));
/// ```
#[allow(mutable_transmutes, unused_mut)]
pub fn eval<'slice, 'node, 'feed, K, T>(
    tensors: &'node [K],
    feeds: &'slice [Feed<'node, 'feed, T>],
) -> Vec<Option<NdArray<T>>>
where
    K: AsRef<Tensor<T>>,
    T: Float,
{
    let mut output_info_store = FxHashMap::<&Tensor<T>, NodeMetadata<T>>::default();
    let mut owned_storage: Vec<NdArray<T>> = Vec::with_capacity(100);

    {
        let mut view_storage: Vec<NdArrayView<T>> = Vec::new();
        let mut feed_store = FxHashMap::<*const Tensor<T>, NdArrayView<'feed, T>>::default();

        let mut dfs_stack = Vec::<(&Tensor<T>, bool)>::with_capacity(100);
        for t in tensors.iter() {
            dfs_stack.push((t.as_ref(), false));
        }

        // Obtain array resources while visiting nodes in topological order.
        // Stack-based depth-first-search is used to avoid stack overflow in explicit recursion.
        while let Some((node, is_parent)) = dfs_stack.pop() {
            let node: &Tensor<T> = node;
            if is_parent {
                // Visit this node
                if node.is_placeholder {
                    let mut found = None;
                    for feed in feeds {
                        if Arc::ptr_eq(feed.0, node) {
                            let clone = feed.1.clone();  // clone the view
                            if !node.known_shape.as_ref().unwrap().validate(clone.shape()) {
                                panic!(
                                    "Shape error: placeholder required {:?}, but got {:?}",
                                    node.known_shape.as_ref().unwrap().get(),
                                    clone.shape()
                                );
                            }
                            found = Some(clone);
                            break;
                        }
                    }
                    unsafe {
                        mem::transmute::<_, &mut FxHashMap<_, _>>(&feed_store)
                            .insert(node as *const _, found.expect("Placeholder unfilled."));
                    }
                } else {
                    if output_info_store.contains_key(node) {
                        continue;
                    }
                    if !node.has_persistent_array {
                        // Aggregate input info for Op::compute
//                        let mut info_list: Vec<ValueInfo<T>> = Vec::with_capacity(node.inputs.len());
//                        let mut contains_no_output = false;
//                        for (in_node, &i) in node.inputs.iter().zip(&node.input_indices) {
//                            let meta = output_info_store.get(in_node).unwrap();
//                            let info = &meta.info_list[i];
//                            if info.ty == ValueType::NoOutput {
//                                contains_no_output = true;
//                                break;
//                            }
//                            info_list.push((*info).clone());  //  "clone" costs little (ValueInfo::Value is always None here)
//                        }

                        // Call Op::compute
                        let ys: op::ComputeResults<_> = {
                            let mut xs = Vec::with_capacity(node.inputs.len());
                            for (in_node, &i) in node.inputs.iter().zip(&node.input_indices) {
                                if in_node.is_placeholder {
                                    let ret: Option<&NdArrayView<T>> = feed_store.get(&(in_node as *const _));
                                    xs.push(Input::Free(ret.unwrap().view()));
                                } else if let Some(ref lock) = in_node.variable_array {
                                    xs.push(Input::RwLock(lock as *const _));
                                } else if let Some(ref arr) = in_node.get_constant_array() {
                                    xs.push(Input::Free(arr.view()));
                                } else {
                                    // Search output of other nodes
                                    let info: &ValueInfo<T> = &output_info_store.get(in_node).unwrap().info_list[i];
                                    match info.ty {
                                        ValueType::Owned => {
                                            xs.push(Input::Free(owned_storage[info.key].view()));
                                        }
                                        ValueType::View => {
                                            // Clone the view
                                            xs.push(Input::Free(view_storage[info.key].clone()));
                                        }
                                        ValueType::NoOutput => {
                                            panic!("{}'s output, which is empty, was fed to {}", in_node.op.name(), node.op.name());
                                        }
                                    }
                                }
                            }
                            node.op.compute(OpComputeContext::new(node, xs))
                        };

                        // Aggregate compute result
                        let mut info_list = Vec::with_capacity(ys.len());
                        let mut contains_no_output = false;
                        for y in ys {
                            match y {
                                Ok(crate::ArrRepr::Owned(val)) => {
                                    info_list.push(ValueInfo::new(
                                        ValueType::Owned,
                                        owned_storage.len(),
                                    ));
                                    unsafe {
                                        // safe
                                        mem::transmute::<_, &mut Vec<_>>(&owned_storage).push(val);
                                    }
                                }
                                Ok(crate::ArrRepr::View(val)) => {
                                    info_list
                                        .push(ValueInfo::new(ValueType::View, view_storage.len()));
                                    view_storage.push(val);
                                }
                                _ => {
                                    info_list.push(ValueInfo::new(ValueType::NoOutput, 0));
                                    contains_no_output = true;
                                }
                            }
                        }
                        output_info_store.insert(
                            node,
                            NodeMetadata {
                                info_list,
                                contains_no_output,
                            },
                        );
                    };
                }
            } else {
                // Update dfs stack
                dfs_stack.push((node, true));
                // Push children if needed
                for child in &node.inputs {
                    if !output_info_store.contains_key(child) {
                        dfs_stack.push((child, false));
                    }
                }
            }
        } // while loop end

        // process array views
        for t in tensors {
            let t = t.as_ref();
            if !t.is_placeholder && !t.has_persistent_array {
                let info: &mut ValueInfo<T> =
                    &mut output_info_store.get_mut(t).unwrap().info_list[0];
                if let ValueType::View = info.ty {
                    info.value = Some(view_storage[info.key].to_owned());
                }
            }
        }
    } // lifetime of views ends here

    for t in tensors {
        let t = t.as_ref();
        if !t.is_placeholder && !t.has_persistent_array {
            let info: &mut ValueInfo<T> = &mut output_info_store.get_mut(t).unwrap().info_list[0];
            if let ValueType::Owned = info.ty {
                info.value = Some(owned_storage[info.key].to_owned());
            }
        }
    }

    let mut ret: Vec<Option<NdArray<T>>> = Vec::with_capacity(tensors.len());
    for t in tensors {
        let t = t.as_ref();
        let arr = if let Some(per) = t.get_persistent_array() {
            // rare case
            Some(per)
        } else if t.is_placeholder {
            // rare case
            let mut found = None;
            for feed in feeds {
                if Arc::ptr_eq(feed.0, t) {
                    found = Some(&feed.1);
                    break;
                }
            }
            Some(found.expect("Placeholder unfilled.").to_owned())
        } else {
            mem::replace(
                &mut output_info_store.get_mut(t).unwrap().info_list[0].value,
                None,
            )
        };
        ret.push(arr);
    }
    ret
}

#[test]
fn test_eval() {
    let ref v = crate::ops::placeholder::<f32>(&[3, 2, 1]);
    let ref z = crate::ops::reduce_sum(&crate::ops::squeeze(v, &[2]), &[0, 1], false);
    let ref g = crate::ops::grad(&[z], &[v]);
    let eval_result = &eval(g, &[Feed(v, crate::ndarray_ext::ones(&[3, 2, 1]).view())])[0];
    assert_eq!(eval_result.as_ref().unwrap().shape(), &[3, 2, 1]);
}

#[test]
fn test_constant_eval() {
    let arr = ndarray::arr1(&[0., 0., 0.]);
    assert_eq!(Some(arr.clone().into_dyn()), crate::variable(arr).eval(&[]));
}

#[test]
fn test_placeholder_eval() {
    let arr = crate::ndarray_ext::ones::<f32>(&[3, 2, 1]);
    let ref v = crate::ops::placeholder(&[3, 2, 1]);
    let eval_result = eval(&[v], &[Feed(v, arr.view())]);
    assert_eq!(eval_result[0], Some(arr));
}
