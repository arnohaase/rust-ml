use blas::daxpy;

use crate::n::calc_utils::chunk_wise_bin_op;
use crate::n::tensor::Tensor;
use crate::n::tracker::BinaryTensorOp;

pub struct BinOpPlus {}
impl BinOpPlus {
    pub fn new() -> BinOpPlus {
        BinOpPlus{}
    }

    pub fn raw_plus(lhs: &Tensor, rhs: &Tensor) -> Tensor {
        if lhs.is_zero() {
            return rhs.clone_with_new_id();
        }
        if rhs.is_zero() {
            return lhs.clone_with_new_id();
        }

        //TODO special handling for one?

        chunk_wise_bin_op(lhs, rhs, Self::raw_plus_chunk)
    }

    fn raw_plus_chunk(lhs: &[f64], rhs: &[f64], result: &mut Vec<f64>) {
        let offs = result.len();
        result.extend_from_slice(lhs);
        unsafe {
            daxpy(lhs.len() as i32, 1.0, rhs, 1, &mut result[offs..], 1);
        }
    }
}
impl BinaryTensorOp for BinOpPlus {
    fn calc(&self, lhs: &Tensor, rhs: &Tensor) -> Tensor {
        assert_eq!(lhs.dimensions(), rhs.dimensions());
        Self::raw_plus(lhs, rhs)
    }

    fn grad(&self, _lhs: &Tensor, lhs_grad: &Option<Tensor>, _rhs: &Tensor, rhs_grad: &Option<Tensor>) -> Option<Tensor> {
        match (lhs_grad, rhs_grad) {
            (None, None) => None,
            (Some(lhs_grad), None) => Some(lhs_grad.clone()),
            (None, Some(rhs_grad)) => Some(rhs_grad.clone()),
            (Some(lhs_grad), Some(rhs_grad)) => Some(Self::raw_plus(lhs_grad, rhs_grad)),
        }
    }
}