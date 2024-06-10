use std::fmt::{Debug, Formatter};

use crate::base::buf::Buf;
use crate::base::element_wise::ElementWise;
use crate::base::tensor::Tensor;

#[derive(Clone)]
pub struct Matrix<const R: usize, const C: usize> {
    buf: Buf,
}
impl <const R: usize, const C: usize> Debug for Matrix<R,C> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Matrix[{}x{}] {}@{}:", R, C, self.id(), self.version())?;
        self.with_data(|b| write!(f, "{:?}", b))
    }
}
impl <const R: usize, const C: usize> Tensor for Matrix<R,C> {
    fn id(&self) -> u32 {
        self.buf.id()
    }
    fn version(&self) -> u32 {
        self.buf.version()
    }

    fn with_data<X>(&self, f: impl FnOnce(&[f64]) -> X) -> X {
        self.buf.with_data(f)
    }
}
impl <const R: usize, const C: usize> Matrix<R,C> {
    pub fn zero() -> Matrix<R,C> {
        Matrix { buf: Buf::zero(R*C) }
    }
}

impl <const R: usize, const C: usize> ElementWise for Matrix<R,C> {
    fn plus(&self, other: &Self) -> Self {
        Matrix { buf: self.buf.plus(&other.buf) }
    }

    fn minus(&self, other: &Self) -> Self {
        Matrix { buf: self.buf.minus(&other.buf) }
    }

    fn mult(&self, other: &Self) -> Self {
        Matrix { buf: self.buf.mult(&other.buf) }
    }

    fn div(&self, other: &Self) -> Self {
        Matrix { buf: self.buf.div(&other.buf) }
    }

    fn plus_equal(&mut self, other: &Self) {
        self.buf.plus_equal(&other.buf)
    }

    fn minus_equal(&mut self, other: &Self) {
        self.buf.minus_equal(&other.buf)
    }

    fn mult_equal(&mut self, other: &Self) {
        self.buf.mult_equal(&other.buf)
    }

    fn div_equal(&mut self, other: &Self) {
        self.buf.div_equal(&other.buf)
    }
}
