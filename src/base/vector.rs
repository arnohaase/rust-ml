use std::fmt::{Debug, Formatter};
use crate::base::buf::Buf;
use crate::base::element_wise::ElementWise;
use crate::base::tensor::Tensor;

#[derive(Clone)]
pub struct Vector {
    len: usize,
    buf: Buf,
}
impl Debug for Vector {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Vector[{}] {}@{}:", self.len(), self.id(), self.version())?;
        self.buf.dump_data(f)
    }
}
impl Tensor for Vector {
    fn id(&self) -> u32 {
        self.buf.id()
    }
    fn version(&self) -> u32 {
        self.buf.version()
    }
}
impl Vector {
    pub fn zero(n: usize) -> Vector {
        Vector { len: n, buf: Buf::zero(n) }
    }

    pub fn len(&self) -> usize {
        self.len
    }
}

impl ElementWise for Vector {
    fn plus(&self, other: &Self) -> Self {
        assert_eq!(self.len(), other.len());
        Vector { len: self.len(), buf: self.buf.plus(&other.buf) }
    }

    fn minus(&self, other: &Self) -> Self {
        assert_eq!(self.len(), other.len());
        Vector { len: self.len(), buf: self.buf.minus(&other.buf) }
    }

    fn mult(&self, other: &Self) -> Self {
        assert_eq!(self.len(), other.len());
        Vector { len: self.len(), buf: self.buf.mult(&other.buf) }
    }

    fn div(&self, other: &Self) -> Self {
        assert_eq!(self.len(), other.len());
        Vector { len: self.len(), buf: self.buf.div(&other.buf) }
    }

    fn plus_equal(&mut self, other: &Self) {
        assert_eq!(self.len(), other.len());
        self.buf.plus_equal(&other.buf)
    }

    fn minus_equal(&mut self, other: &Self) {
        assert_eq!(self.len(), other.len());
        self.buf.minus_equal(&other.buf)
    }

    fn mult_equal(&mut self, other: &Self) {
        assert_eq!(self.len(), other.len());
        self.buf.mult_equal(&other.buf)
    }

    fn div_equal(&mut self, other: &Self) {
        assert_eq!(self.len(), other.len());
        self.buf.div_equal(&other.buf)
    }
}

