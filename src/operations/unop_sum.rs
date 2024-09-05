use blas::{saxpy, sscal};

use crate::tensor::Tensor;
use crate::tensor_env::{BlasEnv, TensorEnv, WgpuEnv};
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
    match dim.num_dims() {
        0 => tensor.clone_with_new_id(),
        1 => {
            // this is an optimization for the important special case of summarizing scalars
            let mut sum = buf.iter().sum();
            if divide_by_len {
                sum /= buf.len() as f32;
            }
            tensor.env().create_tensor(vec![].into(), vec![sum])
        }
        _ => {
            let result_dim = dim.raw()[1..].to_vec();
            let chunk_size = result_dim.iter().map(|d| d.len).product();
            let mut result_buf = buf[0..chunk_size].to_vec();

            for chunk in buf[chunk_size..].chunks(chunk_size) {
                unsafe {
                    saxpy(chunk_size as i32, 1.0, chunk, 1, &mut result_buf, 1);
                }
            }
            if divide_by_len {
                unsafe {
                    sscal(chunk_size as i32, 1.0 / dim.raw()[0].len as f32, &mut result_buf, 1);
                }
            }

            tensor.env().create_tensor(result_dim.into(), result_buf)
        }
    }
}


impl UnaryTensorOp<WgpuEnv> for UnOpSum {
    fn calc<'env>(&self, tensor: &Tensor<'env, WgpuEnv>) -> Tensor<'env, WgpuEnv> {
        if tensor.is_scalar() {
            return tensor.clone_with_new_id()
        }



        todo!()
    }

    fn grad<'env>(&self, t: &Tensor<'env, WgpuEnv>, t_grad: &Option<Tensor<'env, WgpuEnv>>) -> Option<Tensor<'env, WgpuEnv>> {
        todo!()
    }
}




/*

// Constants defining the workgroup size
const WORKGROUP_SIZE: u32 = 64;

// The input buffer containing the array elements
@group(0) @binding(0) var<storage, read> inputArray: array<f32>;

// The output buffer where the final sum will be stored
@group(0) @binding(1) var<storage, read_write> outputSum: f32;

// Shared memory for workgroup reduction
var<workgroup> localSum: array<f32, WORKGROUP_SIZE>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(workgroup_id) workgroup_id: vec3<u32>) {

    // Step 1: Each thread loads its corresponding element
    let index = global_id.x;
    let value = inputArray[index];

    // Step 2: Perform local reduction
    localSum[local_id.x] = value;
    workgroupBarrier();

    // Perform parallel reduction within the workgroup
    var step = WORKGROUP_SIZE / 2;
    while (step > 0) {
        if (local_id.x < step) {
            localSum[local_id.x] += localSum[local_id.x + step];
        }
        step = step / 2;
        workgroupBarrier();
    }

    // Step 3: The first thread in the workgroup writes the local sum to the output buffer
    if (local_id.x == 0) {
        atomicAdd(&outputSum, localSum[0]);
    }
}


 */