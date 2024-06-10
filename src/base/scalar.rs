use std::fmt::{Debug, Formatter};
use crate::base::buf::Buf;
use crate::base::element_wise::ElementWise;
use crate::base::tensor::Tensor;

#[derive(Clone)]
pub struct Scalar {
    buf: Buf,
}
impl Debug for Scalar {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Scalar {}@{}:{}", self.id(), self.version(), self.with_data(|b| b[0]))
    }
}
impl Tensor for Scalar {
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
impl Scalar {
    pub fn new(value: f64) -> Scalar {
        Scalar {
            buf: Buf::from_data(vec![value])
        }
    }
}

impl ElementWise for Scalar {
    fn plus(&self, other: &Self) -> Self {
        Scalar { buf: self.buf.plus(&other.buf) }
    }

    fn minus(&self, other: &Self) -> Self {
        Scalar { buf: self.buf.minus(&other.buf) }
    }

    fn mult(&self, other: &Self) -> Self {
        Scalar { buf: self.buf.mult(&other.buf) }
    }

    fn div(&self, other: &Self) -> Self {
        Scalar { buf: self.buf.div(&other.buf) }
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
