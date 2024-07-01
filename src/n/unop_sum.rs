use blas::{daxpy, dscal};

use crate::n::tensor::Tensor;
use crate::n::tracker::UnaryTensorOp;

pub struct UnOpSum {}
impl UnaryTensorOp for UnOpSum {
    fn calc(&self, tensor: &Tensor) -> Tensor {
        sum_raw(tensor, false)
    }

    fn grad(&self, t: &Tensor, t_grad: &Option<Tensor>) -> Option<Tensor> {
        t_grad.as_ref()
            .map(|grad| sum_raw(grad, false))
    }
}

pub fn sum_raw(tensor: &Tensor, divide_by_len: bool) -> Tensor {
    let dim = tensor.dimensions();
    let buf = &tensor.buf().read().unwrap();
    match dim.len() {
        0 => panic!("called sum() on a scalar"),
        1 => {
            // this is an optimization for the important special case of summarizing scalars
            let mut sum = buf.iter().sum();
            if divide_by_len {
                sum /= buf.len() as f64;
            }
            Tensor::from_raw(vec![], vec![sum])
        }
        _ => {
            let result_dim = dim[1..].to_vec();
            let chunk_size = result_dim.iter().product();
            let mut result_buf = buf[0..chunk_size].to_vec();

            for chunk in buf[chunk_size..].chunks(chunk_size) {
                unsafe {
                    daxpy(chunk_size as i32, 1.0, chunk, 1, &mut result_buf, 1);
                }
            }
            if divide_by_len {
                unsafe {
                    dscal(chunk_size as i32, 1.0 / dim[0] as f64, &mut result_buf, 1);
                }
            }

            Tensor::from_raw(result_dim, result_buf)
        }
    }
}
