use std::fmt::{Debug, Formatter};

use crate::base::buf::Buf;
use crate::base::element_wise::ElementWise;
use crate::base::tensor::Tensor;

#[derive(Clone)]
pub struct Matrix {
    num_rows: usize,
    num_cols: usize,
    buf: Buf,
}
impl Debug for Matrix {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Matrix[{}x{}] {}@{}:", self.num_rows(), self.num_cols(), self.id(), self.version())?;
        self.buf.dump_data(f)
    }
}
impl Tensor for Matrix {
    fn id(&self) -> u32 {
        self.buf.id()
    }
    fn version(&self) -> u32 {
        self.buf.version()
    }
}
impl Matrix {
    pub fn zero(num_rows: usize, num_cols: usize) -> Matrix {
        Matrix { num_rows, num_cols, buf: Buf::zero(num_rows*num_cols) }
    }

    pub fn num_rows(&self) -> usize {
        self.num_rows
    }
    pub fn num_cols(&self) -> usize {
        self.num_cols
    }
}

impl ElementWise for Matrix {
    fn plus(&self, other: &Self) -> Self {
        assert_eq!(self.num_rows(), other.num_rows());
        assert_eq!(self.num_cols(), other.num_cols());
        Matrix { num_rows: self.num_rows, num_cols: self.num_cols, buf: self.buf.plus(&other.buf) }
    }

    fn minus(&self, other: &Self) -> Self {
        assert_eq!(self.num_rows(), other.num_rows());
        assert_eq!(self.num_cols(), other.num_cols());
        Matrix { num_rows: self.num_rows, num_cols: self.num_cols, buf: self.buf.minus(&other.buf) }
    }

    fn mult(&self, other: &Self) -> Self {
        assert_eq!(self.num_rows(), other.num_rows());
        assert_eq!(self.num_cols(), other.num_cols());
        Matrix { num_rows: self.num_rows, num_cols: self.num_cols, buf: self.buf.mult(&other.buf) }
    }

    fn div(&self, other: &Self) -> Self {
        assert_eq!(self.num_rows(), other.num_rows());
        assert_eq!(self.num_cols(), other.num_cols());
        Matrix { num_rows: self.num_rows, num_cols: self.num_cols, buf: self.buf.div(&other.buf) }
    }

    fn plus_equal(&mut self, other: &Self) {
        assert_eq!(self.num_rows(), other.num_rows());
        assert_eq!(self.num_cols(), other.num_cols());
        self.buf.plus_equal(&other.buf)
    }

    fn minus_equal(&mut self, other: &Self) {
        assert_eq!(self.num_rows(), other.num_rows());
        assert_eq!(self.num_cols(), other.num_cols());
        self.buf.minus_equal(&other.buf)
    }

    fn mult_equal(&mut self, other: &Self) {
        assert_eq!(self.num_rows(), other.num_rows());
        assert_eq!(self.num_cols(), other.num_cols());
        self.buf.mult_equal(&other.buf)
    }

    fn div_equal(&mut self, other: &Self) {
        assert_eq!(self.num_rows(), other.num_rows());
        assert_eq!(self.num_cols(), other.num_cols());
        self.buf.div_equal(&other.buf)
    }
}
