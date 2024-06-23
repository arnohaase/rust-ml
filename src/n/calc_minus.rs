use blas::{daxpy, dscal};

use crate::n::tensor::{Repr, Tensor};
use crate::n::tracker::BinaryTensorOp;

pub fn raw_minus(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    BinOpMinus{}.calc(lhs, rhs)
}


pub struct BinOpMinus {}
impl BinOpMinus {
    pub fn new() -> BinOpMinus {
        BinOpMinus{}
    }

    pub fn raw_minus(lhs: &Tensor, rhs: &Tensor) -> Tensor {
        match (lhs.repr(), rhs.repr()) {
            (Repr::ZERO, Repr::BLAS { buf}) => {
                let mut result = buf.read().unwrap().clone();
                unsafe {
                    dscal(result.len() as i32, -1.0, result.as_mut_slice(), 1);
                }
                Tensor::from_raw(lhs.dimensions().into(), result)
            }
            (_, Repr::ZERO) => lhs.clone(),
            (Repr::ONE, _) => todo!(),
            (_, Repr::ONE) => todo!(),
            (Repr::BLAS { buf: lhs_buf }, Repr::BLAS { buf: rhs_buf }) => {
                let lhs_buf = lhs_buf.read().unwrap();
                let rhs_buf = rhs_buf.read().unwrap();

                let mut result = lhs_buf.clone();
                unsafe {
                    daxpy(result.len() as i32, -1.0, rhs_buf.as_slice(), 1, result.as_mut_slice(), 1);
                }
                Tensor::from_raw(lhs.dimensions().into(), result)
            }
        }
    }
}
impl BinaryTensorOp for BinOpMinus {
    fn calc(&self, lhs: &Tensor, rhs: &Tensor) -> Tensor {
        assert_eq!(lhs.dimensions(), rhs.dimensions());
        Self::raw_minus(lhs, rhs)
    }

    fn grad(&self, _lhs: &Tensor, lhs_grad: &Option<Tensor>, _rhs: &Tensor, rhs_grad: &Option<Tensor>) -> Option<Tensor> {
        match (lhs_grad, rhs_grad) {
            (None, None) => None,
            (Some(lhs_grad), None) => Some(lhs_grad.clone()),
            (None, Some(rhs_grad)) => Some(raw_minus(&Tensor::zero(rhs_grad.dimensions().into()), rhs_grad)),
            (Some(lhs_grad), Some(rhs_grad)) => Some(raw_minus(lhs_grad, rhs_grad)),
        }
    }
}