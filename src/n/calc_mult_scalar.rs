use blas::{daxpy, dscal};

use crate::n::tensor::{Repr, Tensor};
use crate::n::tracker::BinaryTensorOp;

pub struct BinOpMultScalar {}
impl BinOpMultScalar {
    pub fn new() -> BinOpMultScalar {
        BinOpMultScalar{}
    }

    pub fn raw_mult_scalar(lhs: &Tensor, rhs_scalar: &Tensor) -> Tensor {
        match (lhs.repr(), rhs_scalar.repr()) {
            (&Repr::ZERO, _) | (_, &Repr::ZERO) => Tensor::zero(lhs.dimensions().into()),
            (&Repr::ONE, _) => todo!("scalar-multiplying tensor"),
            (_, &Repr::ONE) => lhs.clone(),
            (Repr::BLAS { buf: lhs_buf }, Repr::BLAS { buf: rhs_buf }) => {
                let lhs_buf = lhs_buf.read().unwrap();
                let rhs_buf = rhs_buf.read().unwrap();

                let mut result = lhs_buf.clone();
                unsafe {
                    dscal(result.len() as i32, rhs_buf[0], result.as_mut_slice(), 1);
                }
                Tensor::from_raw(lhs.dimensions().into(), result)
            }
        }
    }
}
impl BinaryTensorOp for BinOpMultScalar {
    fn calc(&self, lhs: &Tensor, rhs: &Tensor) -> Tensor {
        assert_eq!(rhs.dimensions(), &[]);
        Self::raw_mult_scalar(lhs, rhs)
    }

    fn grad(&self, _lhs: &Tensor, lhs_grad: &Option<Tensor>, _rhs: &Tensor, rhs_grad: &Option<Tensor>) -> Option<Tensor> {
        match (lhs_grad, rhs_g rad) {
            (None, None) => None,
            (Some(lhs_grad), None) => Some(lhs_grad.clone()),
            (None, Some(rhs_grad)) => Some(rhs_grad.clone()),
            (Some(lhs_grad), Some(rhs_grad)) => Some(Self::raw_plus(lhs_grad, rhs_grad)),
        }
    }
}