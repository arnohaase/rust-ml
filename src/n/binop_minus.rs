use blas::{daxpy, dscal};

use crate::n::calc_utils::chunk_wise_bin_op;
use crate::n::tensor::Tensor;
use crate::n::tracker::BinaryTensorOp;

pub fn raw_minus(lhs: &Tensor, rhs: &Tensor) -> Tensor {
    BinOpMinus{}.calc(lhs, rhs)
}


#[derive(Debug)]
pub struct BinOpMinus {}
impl BinOpMinus {
    pub fn new() -> BinOpMinus {
        BinOpMinus{}
    }

    pub fn raw_minus(lhs: &Tensor, rhs: &Tensor) -> Tensor {
        if rhs.is_zero() {
            return lhs.clone_with_new_id();
        }
        if lhs.is_zero() {
            let mut result = rhs.buf().read().unwrap().clone();
            unsafe {
                dscal(result.len() as i32, -1.0, result.as_mut_slice(), 1);
            }
            return Tensor::from_raw(lhs.dimensions().into(), result);
        }

        chunk_wise_bin_op(lhs, rhs, Self::raw_minus_chunk)
    }

    fn raw_minus_chunk(n: usize, rhs: &[f64], inc_rhs: usize, lhs: &mut[f64], inc_lhs: usize) {
        unsafe {
            daxpy(n as i32, -1.0, rhs, inc_rhs as i32, lhs, inc_lhs as i32);
        }
    }
}
impl BinaryTensorOp for BinOpMinus {
    fn calc(&self, lhs: &Tensor, rhs: &Tensor) -> Tensor {
        Self::raw_minus(lhs, rhs)
    }

    fn grad(&self, _lhs: &Tensor, lhs_grad: &Option<Tensor>, _rhs: &Tensor, rhs_grad: &Option<Tensor>) -> Option<Tensor> {
        match (lhs_grad, rhs_grad) {
            (None, None) => None,
            (Some(lhs_grad), None) => Some(lhs_grad.clone()),
            (None, Some(rhs_grad)) => Some(raw_minus(&Tensor::zero(), rhs_grad)),
            (Some(lhs_grad), Some(rhs_grad)) => Some(raw_minus(lhs_grad, rhs_grad)),
        }
    }
}