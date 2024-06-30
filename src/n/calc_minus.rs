use blas::{daxpy, dscal};

use crate::n::calc_utils::chunk_wise_bin_op;
use crate::n::tensor::Tensor;
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

    fn raw_minus_chunk(lhs: &[f64], rhs: &[f64], result: &mut Vec<f64>) {
        let offs = result.len();
        result.extend_from_slice(lhs);
        unsafe {
            daxpy(lhs.len() as i32, -1.0, rhs, 1, &mut result[offs..], 1);
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
            (None, Some(rhs_grad)) => Some(raw_minus(&Tensor::zero(), rhs_grad)),
            (Some(lhs_grad), Some(rhs_grad)) => Some(raw_minus(lhs_grad, rhs_grad)),
        }
    }
}