use blas::daxpy;

use crate::operations::calc_utils::chunk_wise_bin_op;
use crate::operations::unop_minus::UnOpMinus;
use crate::tensor::Tensor;
use crate::tensor_env::BlasEnv;
use crate::tracker::BinaryTensorOp;

pub fn raw_minus<'env>(lhs: &Tensor<'env, BlasEnv>, rhs: &Tensor<'env, BlasEnv>) -> Tensor<'env, BlasEnv> {
    BinOpMinus{}.calc(lhs, rhs)
}


#[derive(Debug)]
pub struct BinOpMinus {}
impl BinOpMinus {
    pub fn new() -> BinOpMinus {
        BinOpMinus{}
    }

    pub fn raw_minus<'env>(lhs: &Tensor<'env, BlasEnv>, rhs: &Tensor<'env, BlasEnv>) -> Tensor<'env, BlasEnv> {
        // if rhs.is_zero() {
        //     return lhs.clone_with_new_id();
        // }
        // if lhs.is_zero() {
        //     let mut result = rhs.buf().read().unwrap().clone();
        //     unsafe {
        //         dscal(result.len() as i32, -1.0, result.as_mut_slice(), 1);
        //     }
        //     return Tensor::from_raw(lhs.dimensions().into(), result);
        // }
        chunk_wise_bin_op(lhs, rhs, false, Self::raw_minus_chunk)
    }

    fn raw_minus_chunk(n: usize, rhs: &[f64], inc_rhs: usize, lhs: &mut[f64], inc_lhs: usize) {
        unsafe {
            daxpy(n as i32, -1.0, rhs, inc_rhs as i32, lhs, inc_lhs as i32);
        }
    }
}
impl BinaryTensorOp<BlasEnv> for BinOpMinus {
    fn calc<'env>(&self, lhs: &Tensor<'env, BlasEnv>, rhs: &Tensor<'env, BlasEnv>) -> Tensor<'env, BlasEnv> {
        Self::raw_minus(lhs, rhs)
    }

    fn grad<'env>(&self, _lhs: &Tensor<'env, BlasEnv>, lhs_grad: &Option<Tensor<'env, BlasEnv>>, _rhs: &Tensor<'env, BlasEnv>, rhs_grad: &Option<Tensor<'env, BlasEnv>>) -> Option<Tensor<'env, BlasEnv>> {
        match (lhs_grad, rhs_grad) {
            (None, None) => None,
            (Some(lhs_grad), None) => Some(lhs_grad.clone()),
            (None, Some(rhs_grad)) => Some(UnOpMinus::raw_minus(rhs_grad)),
            (Some(lhs_grad), Some(rhs_grad)) => Some(raw_minus(lhs_grad, rhs_grad)),
        }
    }
}