use crate::n::tensor::Tensor;
use crate::n::tracker::UnaryTensorOp;
use crate::n::unop_sum::sum_raw;

#[derive(Debug)]
pub struct UnOpAvg {}
impl UnaryTensorOp for UnOpAvg {
    fn calc(&self, tensor: &Tensor) -> Tensor {
        sum_raw(tensor, true)
    }

    fn grad(&self, _t: &Tensor, t_grad: &Option<Tensor>) -> Option<Tensor> {
        t_grad.as_ref()
            .map(|grad| sum_raw(grad, true))
    }
}
