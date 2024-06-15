use std::cell::RefCell;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::rc::Rc;

use crate::base::buf::new_tensor_id;
use crate::base::tensor::Tensor;

#[derive(Clone, Debug)]
pub struct Collection<T: Tensor> {
    id: u32,
    version: Rc<RefCell<u32>>,
    buf: Rc<RefCell<Vec<T>>>,
}
impl <T: Tensor> Collection<T> {
    pub fn new(data: Vec<T>) -> Collection<T> { //TODO from iter
        Collection {
            id: new_tensor_id(),
            version: Rc::new(RefCell::new(0)),
            buf: Rc::new(RefCell::new(data)),
        }
    }
}
impl <T: Tensor> Tensor for Collection<T> {
    fn id(&self) -> u32 {
        self.id
    }

    fn version(&self) -> u32 {
        *self.version.borrow()
    }
}
impl <T: Tensor> CollectionLike<T> for Collection<T> {}

pub trait CollectionLike<T: Tensor>: Clone+Debug {
    fn plus_element_wise<O: CollectionLike<T>>(&self, other: O) -> impl CollectionLike<T> {
        CollectionElementwiseSum::new(self.clone(), other.clone())
    }
}

#[derive(Clone, Debug)]
pub struct CollectionElementwiseSum<T: Tensor, L: CollectionLike<T>, R: CollectionLike<T>> {
    id: u32,
    lhs: L,
    rhs: R,
    pd: PhantomData<T>,
}
impl <T: Tensor, L: CollectionLike<T>, R: CollectionLike<T>> CollectionElementwiseSum<T, L, R> {
    fn new(lhs: L, rhs: R) -> CollectionElementwiseSum<T, L, R> {
        CollectionElementwiseSum {
            id: new_tensor_id(),
            lhs,
            rhs,
            pd: Default::default(),
        }
    }
}
impl <T: Tensor, L: CollectionLike<T>, R: CollectionLike<T>> CollectionLike<T> for CollectionElementwiseSum<T, L, R> {}
impl <T: Tensor, L: CollectionLike<T>, R: CollectionLike<T>> Tensor for CollectionElementwiseSum<T, L, R> {
    fn id(&self) -> u32 {
        self.id
    }
    fn version(&self) -> u32 {
        0
    }
}

