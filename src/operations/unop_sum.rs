use blas::{saxpy, sscal};

use crate::tensor::Tensor;
use crate::tensor_env::{BlasEnv, TensorEnv};
use crate::tracker::UnaryTensorOp;

#[derive(Debug)]
pub struct UnOpSum {}
impl UnaryTensorOp<BlasEnv> for UnOpSum {
    fn calc<'env>(&self, tensor: &Tensor<'env, BlasEnv>) -> Tensor<'env, BlasEnv> {
        sum_raw(tensor, false)
    }

    fn grad<'env>(&self, _t: &Tensor<'env, BlasEnv>, t_grad: &Option<Tensor<'env, BlasEnv>>) -> Option<Tensor<'env, BlasEnv>> {
        t_grad.as_ref()
            .map(|grad| sum_raw(grad, false))
    }
}

pub fn sum_raw<'env>(tensor: &Tensor<'env, BlasEnv>, divide_by_len: bool) -> Tensor<'env, BlasEnv> {
    let dim = tensor.dimensions();
    //TODO verify that the outermost dimension has kind 'collection'? Generalize to sum on a selectable dimension? With assertable kind?
    let buf = &tensor.buf().read().unwrap();
    match dim.len() {
        0 => panic!("called sum() on a scalar"),
        1 => {
            // this is an optimization for the important special case of summarizing scalars
            let mut sum = buf.iter().sum();
            if divide_by_len {
                sum /= buf.len() as f32;
            }
            tensor.env().create_tensor(vec![], vec![sum])
        }
        _ => {
            let result_dim = dim[1..].to_vec();
            let chunk_size = result_dim.iter().map(|d| d.len).product();
            let mut result_buf = buf[0..chunk_size].to_vec();

            for chunk in buf[chunk_size..].chunks(chunk_size) {
                unsafe {
                    saxpy(chunk_size as i32, 1.0, chunk, 1, &mut result_buf, 1);
                }
            }
            if divide_by_len {
                unsafe {
                    sscal(chunk_size as i32, 1.0 / dim[0].len as f32, &mut result_buf, 1);
                }
            }

            tensor.env().create_tensor(result_dim, result_buf)
        }
    }
}
