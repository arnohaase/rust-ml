use blas::sscal;
use crate::tensor::Tensor;
use crate::tensor_env::{BlasEnv, TensorEnv};
use crate::tracker::UnaryTensorOp;

#[derive(Debug)]
pub struct UnOpMinus {}

impl UnOpMinus {
    pub fn raw_minus<'env>(tensor: &Tensor<'env, BlasEnv>) -> Tensor<'env, BlasEnv> {
        let mut result = tensor.buf().read().unwrap().clone();
        unsafe {
            sscal(result.len() as i32, -1.0, result.as_mut_slice(), 1);
        }
        tensor.env().create_tensor(tensor.dimensions().into(), result)
    }
}

impl UnaryTensorOp<BlasEnv> for UnOpMinus {
    fn calc<'env>(&self, tensor: &Tensor<'env, BlasEnv>) -> Tensor<'env, BlasEnv> {
        Self::raw_minus(tensor)
    }

    fn grad<'env>(&self, _t: &Tensor<'env, BlasEnv>, t_grad: &Option<Tensor<'env, BlasEnv>>) -> Option<Tensor<'env, BlasEnv>> {
        if let Some(grad) = t_grad {
            return Some(Self::raw_minus(grad));
        }
        None
    }
}