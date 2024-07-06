use blas::dscal;

use crate::n::binop_plus::BinOpPlus;
use crate::n::calc_utils::chunk_wise_bin_op;
use crate::n::tensor::Tensor;
use crate::n::tracker::BinaryTensorOp;

pub struct BinOpMult {}
impl BinOpMult {
    pub fn new() -> BinOpMult {
        BinOpMult{}
    }

    pub fn raw_mult(lhs: &Tensor, rhs: &Tensor) -> Tensor {
        if lhs.is_zero() || rhs.is_zero() {
            return Tensor::zero();
        }
        if lhs.is_one() {
            return rhs.clone_with_new_id();
        }
        if rhs.is_one() {
            return lhs.clone_with_new_id();
        }

        if lhs.is_scalar() {
            return Self::raw_mult_scalar(lhs, rhs);
        }
        if rhs.is_scalar() {
            return Self::raw_mult_scalar(rhs, lhs);
        }

        chunk_wise_bin_op(lhs, rhs, Self::raw_mult_chunk)
    }

    fn raw_mult_scalar(scalar: &Tensor, regular: &Tensor) -> Tensor {
        let scalar = scalar.buf().read().unwrap()[0];
        let mut result = regular.buf().read().unwrap().to_vec();
        unsafe {
            dscal(result.len() as i32, scalar, result.as_mut_slice(), 1);
        }
        return Tensor::from_raw(regular.dimensions().into(), result);
    }

    fn raw_mult_chunk(lhs: &[f64], rhs: &[f64], result: &mut Vec<f64>) {
        for i in 0..lhs.len() {
            result.push(lhs[i] * rhs[i]);
        }
    }
}
impl BinaryTensorOp for BinOpMult {
    fn calc(&self, lhs: &Tensor, rhs: &Tensor) -> Tensor {
        Self::raw_mult(lhs, rhs)
    }

    fn grad(&self, lhs: &Tensor, lhs_grad: &Option<Tensor>, rhs: &Tensor, rhs_grad: &Option<Tensor>) -> Option<Tensor> {
        match (lhs_grad, rhs_grad) {
            (None, None) => None,
            (Some(lhs_grad), None) => Some(Self::raw_mult(lhs_grad, rhs)),
            (None, Some(rhs_grad)) => Some(Self::raw_mult(lhs, rhs_grad)),
            (Some(lhs_grad), Some(rhs_grad)) => {
                Some(BinOpPlus::raw_plus(
                    &Self::raw_mult(lhs_grad, rhs),
                    &Self::raw_mult(lhs, rhs_grad),
                ))
            },
        }
    }
}