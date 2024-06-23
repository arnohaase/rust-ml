use std::cell::RefCell;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::rc::Rc;

use crate::base::tensor::Tensor;

#[derive(Clone, Debug)]
pub struct Collection<T: Tensor> {
    buf: Rc<RefCell<Vec<T>>>,
}
impl <T: Tensor> Collection<T> {
    pub fn new(data: Vec<T>) -> Collection<T> {
        Collection {
            buf: Rc::new(RefCell::new(data)),
        }
    }
}
impl <T: Tensor, I: Iterator<Item=T>> From<I> for Collection<T> {
    fn from(iter: I) -> Self {
        Self::new(iter.collect())
    }
}

impl <T: Tensor> CollectionLike<T> for Collection<T> {
    fn materialize(self) -> Collection<T> {
        self
    }

    fn get(&self, index: usize) -> Option<T> {
        self.buf.borrow().get(index)
            .map(|o| o.clone())
    }

    fn len(&self) -> usize {
        self.buf.borrow().len()
    }
}

#[derive(Clone, Debug)]
pub struct CollectionElementwiseSum<T: Tensor, L: CollectionLike<T>, R: CollectionLike<T>> {
    lhs: L,
    rhs: R,
    pd: PhantomData<T>,
}
impl <T: Tensor, L: CollectionLike<T>, R: CollectionLike<T>> CollectionElementwiseSum<T, L, R> {
    fn new(lhs: L, rhs: R) -> CollectionElementwiseSum<T, L, R> {
        debug_assert_eq!(lhs.len(), rhs.len());
        CollectionElementwiseSum {
            lhs,
            rhs,
            pd: Default::default(),
        }
    }
}
impl <T: Tensor, L: CollectionLike<T>, R: CollectionLike<T>> CollectionLike<T> for CollectionElementwiseSum<T, L, R> {
    fn get(&self, index: usize) -> Option<T> {
        if let Some(lhs) = self.lhs.get(index) {
            if let Some(rhs) = self.rhs.get(index) {
                return Some(lhs.plus(&rhs))
            }
        }
        None
    }

    fn len(&self) -> usize {
        self.lhs.len()
    }
}

pub trait CollectionLike<T: Tensor>: Clone + Debug {
    fn materialize(self) -> Collection<T> {
        Collection::new(self.iter().collect())
    }

    fn iter(&self) -> CollectionIter<'_, T, Self> {
        CollectionIter {
            coll: self,
            next_index: 0,
            pd: Default::default(),
        }
    }

    fn get(&self, index: usize) -> Option<T>;

    fn len(&self) -> usize;

    fn plus<O: CollectionLike<T>>(&self, other: O) -> impl CollectionLike<T> {
        CollectionElementwiseSum::new(self.clone(), other.clone())
    }

    fn reduce_sum(&self) -> T {
        let mut iter = self.iter();
        let mut result = iter.next()
            .expect("reducing empty collectin");
        for item in iter {
            result = result.plus(&item);
        }
        result
    }
}

pub struct CollectionIter<'a, T: Tensor, X: CollectionLike<T>> {
    coll: &'a X,
    next_index: usize,
    pd: PhantomData<T>,
}
impl <'a, T: Tensor, X: CollectionLike<T>> Iterator for CollectionIter<'a, T, X> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        let result = self.coll.get(self.next_index);
        if result.is_some() {
            self.next_index += 1;
        }
        result
    }
}
