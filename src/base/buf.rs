use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::{AtomicU32, Ordering};

use blas::daxpy;
use lazy_static::lazy_static;

use crate::base::element_wise::ElementWise;

lazy_static! {
    static ref TENSOR_ID_COUNTER: AtomicU32 = AtomicU32::new(0);
}


#[derive(Clone)]
pub struct Buf {
    id: u32,
    version: Rc<RefCell<u32>>,
    data: Rc<RefCell<Vec<f64>>>,
}
impl Buf {
    pub(crate) fn from_data(data: Vec<f64>) -> Buf {
        Buf {
            id: TENSOR_ID_COUNTER.fetch_add(1, Ordering::SeqCst),
            version: Rc::new(RefCell::new(0)),
            data: Rc::new(RefCell::new(data)),
        }
    }

    pub fn zero(size: usize) -> Buf {
        Self::from_data(vec![0.0; size])
    }

    pub fn id(&self) -> u32 {
        self.id
    }
    pub fn version(&self) -> u32 {
        *self.version.borrow()
    }
    pub fn with_data<X>(&self, f: impl FnOnce(&[f64]) -> X) -> X {
        f(self.data.borrow().as_ref())
    }
}
impl ElementWise for Buf {
    fn plus(&self, other: &Self) -> Self {
        let mut result = self.data.borrow().clone();
        unsafe {
            daxpy(result.len() as i32, 1.0, other.data.borrow().as_ref(), 1, result.as_mut_slice(), 1);
        }
        Self::from_data(result)
    }

    fn minus(&self, other: &Self) -> Self {
        let mut result = self.data.borrow().clone();
        unsafe {
            daxpy(result.len() as i32, -1.0, other.data.borrow().as_ref(), 1, result.as_mut_slice(), 1);
        }
        Self::from_data(result)
    }

    fn mult(&self, other: &Self) -> Self {
        let mut result = self.data.borrow().clone();
        let o = other.data.borrow();
        for i in 0..result.len() {
            result[i] *= o[i];
        }
        Self::from_data(result)
    }

    fn div(&self, other: &Self) -> Self {
        let mut result = self.data.borrow().clone();
        let o = other.data.borrow();
        for i in 0..result.len() {
            result[i] /= o[i];
        }
        Self::from_data(result)
    }

    fn plus_equal(&mut self, other: &Self) {
        let mut x = self.data.borrow_mut();
        unsafe {
            daxpy(x.len() as i32, 1.0, other.data.borrow().as_ref(), 1, x.as_mut_slice(), 1);
        }
        *self.version.borrow_mut() += 1;
    }

    fn minus_equal(&mut self, other: &Self) {
        let mut x = self.data.borrow_mut();
        unsafe {
            daxpy(x.len() as i32, -1.0, other.data.borrow().as_ref(), 1, x.as_mut_slice(), 1);
        }
        *self.version.borrow_mut() += 1
    }

    fn mult_equal(&mut self, other: &Self) {
        let mut x = self.data.borrow_mut();
        let o = other.data.borrow();
        for i in 0..x.len() {
            x[i] *= o[i];
        }
        *self.version.borrow_mut() += 1;
    }

    fn div_equal(&mut self, other: &Self) {
        let mut x = self.data.borrow_mut();
        let o = other.data.borrow();
        for i in 0..x.len() {
            x[i] /= o[i];
        }
        *self.version.borrow_mut() += 1;
    }
}
