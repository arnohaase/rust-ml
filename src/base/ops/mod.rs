use crate::base::matrix::Matrix;
use crate::base::scalar::Scalar;
use crate::base::vector::Vector;

mod arith;


pub enum Expression {
    BinOpScalar { lhs: Scalar, rhs: Scalar, op: Box<dyn BinOpScalar> },
    BinOpVVV { lhs: Vector, rhs: Vector, op: Box<dyn BinOpVvv> },
    BinOpVVS { lhs: Vector, rhs: Vector, op: Box<dyn BinOpVvs> },
    BinOpVVM { lhs: Vector, rhs: Vector, op: Box<dyn BinOpVvm> },
    BinOpVSV { lhs: Vector, rhs: Scalar, op: Box<dyn BinOpVsv> },
    BinOpMVV { lhs: Matrix, rhs: Vector, op: Box<dyn BinOpMvv> },
    BinOpMmm { lhs: Matrix, rhs: Matrix, op: Box<dyn BinOpMmm> },
    //TODO reduce
    //TODO unop
}

pub trait BinOpScalar {
    fn calc(&self, lhs: Scalar, rhs: Scalar) -> Scalar;
}

pub trait BinOpVvv {
    fn calc(&self, lhs: Vector, rhs: Vector) -> Vector;
}

pub trait BinOpVvs {
    fn calc(&self, lhs: Vector, rhs: Vector) -> Scalar;
}

pub trait BinOpVvm {
    fn calc(&self, lhs: Vector, rhs: Vector) -> Matrix;
}

pub trait BinOpVsv {
    fn calc(&self, lhs: Vector, rhs: Scalar) -> Vector;
}

pub trait BinOpMvv {
    fn calc(&self, lhs: Matrix, rhs: Vector) -> Vector;
}

pub trait BinOpMmm {
    fn calc(&self, lhs: Matrix, rhs: Matrix) -> Matrix;
}
