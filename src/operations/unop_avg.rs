use crate::operations::unop_sum::sum_raw;
use crate::tensor::Tensor;
use crate::tensor_env::BlasEnv;
use crate::tracker::UnaryTensorOp;

#[derive(Debug)]
pub struct UnOpAvg {}
impl UnaryTensorOp<BlasEnv> for UnOpAvg {
    fn calc<'env>(&self, tensor: &Tensor<'env, BlasEnv>) -> Tensor<'env, BlasEnv> {
        sum_raw(tensor, true)
    }

    fn grad<'env>(&self, _t: &Tensor<'env, BlasEnv>, t_grad: &Option<Tensor<'env, BlasEnv>>) -> Option<Tensor<'env, BlasEnv>> {
        t_grad.as_ref()
            .map(|grad| sum_raw(grad, true))
    }
}
