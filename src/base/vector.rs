use std::fmt::{Debug, Formatter};
use crate::base::buf::Buf;
use crate::base::element_wise::ElementWise;
use crate::base::tensor::Tensor;

#[derive(Clone)]
pub struct Vector<const N: usize> {
    buf: Buf,
}
impl <const N: usize> Debug for Vector<N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Vector[{}] {}@{}:", N, self.id(), self.version())?;
        self.with_data(|b| write!(f, "{:?}", b))
    }
}
impl <const N: usize> Tensor for Vector<N> {
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
impl <const N: usize> Vector<N> {
    pub fn zero() -> Vector<N> {
        Vector { buf: Buf::zero(N) }
    }
}

impl <const N: usize> ElementWise for Vector<N> {
    fn plus(&self, other: &Self) -> Self {
        Vector { buf: self.buf.plus(&other.buf) }
    }

    fn minus(&self, other: &Self) -> Self {
        Vector { buf: self.buf.minus(&other.buf) }
    }

    fn mult(&self, other: &Self) -> Self {
        Vector { buf: self.buf.mult(&other.buf) }
    }

    fn div(&self, other: &Self) -> Self {
        Vector { buf: self.buf.div(&other.buf) }
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

