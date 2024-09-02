use blas::saxpy;
use triomphe::Arc;
use crate::dimension::MatchDimensionsResult;

use crate::operations::calc_utils_blas::chunk_wise_bin_op;
use crate::tensor::Tensor;
use crate::tensor_env::{BlasEnv, WgpuEnv};
use crate::tracker::BinaryTensorOp;

#[derive(Debug)]
pub struct BinOpPlus {}
impl BinOpPlus {
    pub fn new() -> BinOpPlus {
        BinOpPlus{}
    }

    pub fn plus_in_place<'env>(lhs: &mut Tensor<'env, BlasEnv>, rhs: &Tensor<'env, BlasEnv>, factor: f32) {
        // if rhs.is_zero() {
        //     return;
        // }

        let mut lhs_buf = lhs.buf().write().unwrap();
        let rhs_buf = rhs.buf().read().unwrap();
        match lhs.dimensions().match_with_other(rhs.dimensions()) {
            MatchDimensionsResult::Mismatch => todo!("dimension mismatch"),
            MatchDimensionsResult::Equal =>
                unsafe {
                    saxpy(lhs_buf.len() as i32, factor, &rhs_buf, 1, &mut lhs_buf, 1);
                }
            MatchDimensionsResult::LeftContainsRight { num_wrapper_dims, num_nested_dims } => {
                //TODO extract to 'Dimensions' data type
                let chunk_size = lhs.dimensions().size_without_outer(num_wrapper_dims);
                let num_interleaved = lhs.dimensions().size_inner(num_nested_dims);
                for lhs_chunk in lhs_buf.chunks_mut(chunk_size) {
                    for offset in 0..num_interleaved {
                        unsafe {
                            saxpy(chunk_size as i32, factor, &rhs_buf, 1, &mut lhs_chunk[offset..], num_interleaved as i32);
                        }
                    }
                }
            }
            MatchDimensionsResult::RightContainsLeft { num_wrapper_dims, num_nested_dims } => {
                todo!()
            }
        }
    }

    fn raw_plus_chunk_in_place(lhs: &mut [f32], rhs: &[f32]) {
        unsafe {
            saxpy(lhs.len() as i32, -1.0, rhs, 1, lhs, 1);
        }
    }


    pub fn raw_plus<'env>(lhs: &Tensor<'env, BlasEnv>, rhs: &Tensor<'env, BlasEnv>) -> Tensor<'env, BlasEnv> {
        // if lhs.is_zero() {
        //     return rhs.clone_with_new_id();
        // }
        // if rhs.is_zero() {
        //     return lhs.clone_with_new_id();
        // }

        //TODO special handling for Tensor::one?

        chunk_wise_bin_op(lhs, rhs, true, Self::raw_plus_chunk)
    }

    fn raw_plus_chunk(n: usize, rhs: &[f32], inc_rhs: usize, lhs: &mut [f32], inc_lhs: usize) {
        unsafe {
            saxpy(n as i32, 1.0, rhs, inc_rhs as i32, lhs, inc_lhs as i32);
        }
    }
}
impl BinaryTensorOp<BlasEnv> for BinOpPlus {
    fn calc<'env>(&self, lhs: &Tensor<'env, BlasEnv>, rhs: &Tensor<'env, BlasEnv>) -> Tensor<'env, BlasEnv> {
        Self::raw_plus(lhs, rhs)
    }

    fn grad<'env>(&self, _lhs: &Tensor<'env, BlasEnv>, lhs_grad: &Option<Tensor<'env, BlasEnv>>, _rhs: &Tensor<'env, BlasEnv>, rhs_grad: &Option<Tensor<'env, BlasEnv>>) -> Option<Tensor<'env, BlasEnv>> {
        match (lhs_grad, rhs_grad) {
            (None, None) => None,
            (Some(lhs_grad), None) => Some(lhs_grad.clone_with_new_id()),
            (None, Some(rhs_grad)) => Some(rhs_grad.clone_with_new_id()),
            (Some(lhs_grad), Some(rhs_grad)) => Some(Self::raw_plus(lhs_grad, rhs_grad)),
        }
    }
}

impl BinaryTensorOp<WgpuEnv> for BinOpPlus {
    fn calc<'env>(&self, lhs: &Tensor<'env, WgpuEnv>, rhs: &Tensor<'env, WgpuEnv>) -> Tensor<'env, WgpuEnv> {
        match lhs.dimensions().match_with_other(rhs.dimensions()) {
            MatchDimensionsResult::Mismatch => todo!("dimension mismatch"),
            MatchDimensionsResult::Equal => plus_wgpu_left_interleaved_dim(lhs, rhs, 1, lhs.buf().size() as usize, 1),
            MatchDimensionsResult::LeftContainsRight { num_wrapper_dims, num_nested_dims } => {
                let num_chunks = lhs.dimensions().size_outer(num_wrapper_dims);
                let chunk_size = lhs.dimensions().size_without_outer(num_wrapper_dims);
                let num_interleaved: usize = lhs.dimensions().size_inner(num_nested_dims);
                plus_wgpu_left_interleaved_dim(lhs, rhs, num_chunks, chunk_size, num_interleaved)
            },
            MatchDimensionsResult::RightContainsLeft { num_wrapper_dims, num_nested_dims } => {
                let num_chunks = rhs.dimensions().size_outer(num_wrapper_dims);
                let chunk_size = rhs.dimensions().size_without_outer(num_wrapper_dims);
                let num_interleaved: usize = rhs.dimensions().size_inner(num_nested_dims);
                plus_wgpu_left_interleaved_dim(rhs, lhs, num_chunks, chunk_size, num_interleaved)
            }
        }
    }

    fn grad<'env>(&self, _lhs: &Tensor<'env, WgpuEnv>, lhs_grad: &Option<Tensor<'env, WgpuEnv>>, _rhs: &Tensor<'env, WgpuEnv>, rhs_grad: &Option<Tensor<'env, WgpuEnv>>) -> Option<Tensor<'env, WgpuEnv>> {
        match (lhs_grad, rhs_grad) {
            (None, None) => None,
            (Some(lhs_grad), None) => Some(lhs_grad.clone_with_new_id()),
            (None, Some(rhs_grad)) => Some(rhs_grad.clone_with_new_id()),
            (Some(lhs_grad), Some(rhs_grad)) => Some(self.calc(lhs_grad, rhs_grad)),
        }
    }
}

fn plus_wgpu_left_interleaved_dim<'env>(lhs: &Tensor<'env, WgpuEnv>, rhs: &Tensor<'env, WgpuEnv>, num_chunks: usize, chunk_size: usize, interleave_size: usize) -> Tensor<'env, WgpuEnv> {
    let shader = lhs.env().shader(&format!("+:{interleave_size}:{chunk_size}"), || include_str!("binop_plus.wgsl")
        .replace("{n}", &interleave_size.to_string())
        .replace("{chunk_size}", &chunk_size.to_string())
    );

    let result_buf = lhs.env().create_storage_buffer(lhs.buf().size());

    let bind_group = shader.bind_group(lhs.env(), vec![
        lhs.buf().as_entire_binding(),
        rhs.buf().as_entire_binding(),
        result_buf.as_entire_binding(),
    ]);

    let mut encoder = lhs.env().device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(&shader.id) });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(&shader.id),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&shader.compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(1, num_chunks as u32, 1);
    }

    //TODO split chunk parallelism inside and across workgroups

    lhs.env().queue.submit(Some(encoder.finish()));

    Tensor::create_from_raw(lhs.env(), lhs.dimensions().clone(), Arc::new(result_buf))
}

#[cfg(test)]
mod test {
    use rstest::rstest;
    use crate::operations::binop_plus::BinOpPlus;
    use crate::test_utils::tensor_factories::tensor_from_spec;
    use crate::tracker::BinaryTensorOp;
    use crate::with_all_envs;

    #[rstest]
    #[case::scalar("1.0", "2.0", "3.0")]
    #[case::simple_vec("R:[1, 2, 3]", "R:[4, 5, 6]", "R:[5, 7, 9]")]
    #[case::nested_left("R-P:[[1,2][3,4]]", "R:[5,6]", "R-P:[[6,7][9,10]]")]
    #[case::nested_right("R:[5,6]", "R-P:[[1,2][3,4]]", "R-P:[[6,7][9,10]]")]
    #[case::collection_left("C-R:[[1,2,3][4,5,6]]", "R:[2,3,4]", "C-R:[[3,5,7][6,8,10]]")]
    #[case::collection_right("R:[2,3,4]", "C-R:[[1,2,3][4,5,6]]", "C-R:[[3,5,7][6,8,10]]")]
    #[case::both_left("C-R-P:[[[1,2,3]][[4,5,6]]]", "R:[.5]", "C-R-P:[[[1.5,2.5,3.5]][[4.5,5.5,6.5]]]")]
    #[case::both_right("R:[.5]", "C-R-P:[[[1,2,3]][[4,5,6]]]", "C-R-P:[[[1.5,2.5,3.5]][[4.5,5.5,6.5]]]")]
    fn test_add(#[case] a: &str, #[case] b: &str, #[case] expected: &str) {
        with_all_envs!(env => {
            let a = tensor_from_spec(a, &env);
            let b = tensor_from_spec(b, &env);
            let c = BinOpPlus{}.calc(&a, &b);

            c.assert_pretty_much_equal_to(&tensor_from_spec(expected, &env));
        })
    }

    //TODO test grad
}
