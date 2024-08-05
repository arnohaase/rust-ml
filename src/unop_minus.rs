use blas::dscal;
use crate::tensor::Tensor;
use crate::tracker::UnaryTensorOp;

#[derive(Debug)]
pub struct UnOpMinus {}

impl UnOpMinus {
    pub fn raw_minus(tensor: &Tensor) -> Tensor {
        let mut result = tensor.buf().read().unwrap().clone();
        unsafe {
            dscal(result.len() as i32, -1.0, result.as_mut_slice(), 1);
        }
        Tensor::from_raw(tensor.dimensions().into(), result)
    }
}

impl UnaryTensorOp for UnOpMinus {
    fn calc(&self, tensor: &Tensor) -> Tensor {
        Self::raw_minus(tensor)
    }

    fn grad(&self, _t: &Tensor, t_grad: &Option<Tensor>) -> Option<Tensor> {
        if let Some(grad) = t_grad {
            return Some(Self::raw_minus(grad));
        }
        None
    }
}