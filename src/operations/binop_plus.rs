use std::borrow::Cow;
use blas::saxpy;
use triomphe::Arc;

use crate::operations::calc_utils::{chunk_wise_bin_op, fit_dimensions, FitDimensionsResult};
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
        match fit_dimensions(lhs.dimensions(), rhs.dimensions()) {
            FitDimensionsResult::Mismatch => todo!("dimension mismatch"),
            FitDimensionsResult::Equal =>
                unsafe {
                    saxpy(lhs_buf.len() as i32, factor, &rhs_buf, 1, &mut lhs_buf, 1);
                }
            FitDimensionsResult::LeftContainsRight { num_wrapper_dims, num_nested_dims } => {
                //TODO extract to 'Dimensions' data type
                let chunk_size = lhs.dimensions()[num_wrapper_dims..].iter().map(|d| d.len).product(); // empty --> 1
                let num_interleaved: usize = lhs.dimensions()[lhs.dimensions().len() - num_nested_dims..].iter().map(|d| d.len).product();
                for lhs_chunk in lhs_buf.chunks_mut(chunk_size) {
                    for offset in 0..num_interleaved {
                        unsafe {
                            saxpy(chunk_size as i32, factor, &rhs_buf, 1, &mut lhs_chunk[offset..], num_interleaved as i32);
                        }
                    }
                }
            }
            FitDimensionsResult::RightContainsLeft { num_wrapper_dims, num_nested_dims } => {
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
            (Some(lhs_grad), None) => Some(lhs_grad.clone()),
            (None, Some(rhs_grad)) => Some(rhs_grad.clone()),
            (Some(lhs_grad), Some(rhs_grad)) => Some(Self::raw_plus(lhs_grad, rhs_grad)),
        }
    }
}

impl BinaryTensorOp<WgpuEnv> for BinOpPlus {
    fn calc<'env>(&self, lhs: &Tensor<'env, WgpuEnv>, rhs: &Tensor<'env, WgpuEnv>) -> Tensor<'env, WgpuEnv> {
        assert_eq!(lhs.dimensions(), rhs.dimensions()); //TODO

        //TODO f32
        //TODO workgroup size
        let wgsl = r#"
            struct Data {
                values: array<f32>,
            };

            @group(0) @binding(0) var<storage, read> a: Data;
            @group(0) @binding(1) var<storage, read> b: Data;
            @group(0) @binding(2) var<storage, read_write> result: Data;

            @compute @workgroup_size(32)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                let index = id.x;
                result.values[index] = a.values[index] + b.values[index];
            }
        "#;

        //TODO caching
        let module = lhs.env().device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("BinOpPlus"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(wgsl)),
        });
        //TODO caching
        let compute_pipeline = lhs.env().device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None, //TODO
            layout: None,
            module: &module,
            entry_point: "main",
            compilation_options: Default::default(),
            cache: None,
        });

        let result_buf = lhs.env().create_storage_buffer(lhs.buf().size());

        let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
        let bind_group = lhs.env().device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, //TODO
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: lhs.buf().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: rhs.buf().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: result_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = lhs.env().device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None }); //TODO label
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None, //TODO label
                timestamp_writes: None,
            });
            cpass.set_pipeline(&compute_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(lhs.buf().size() as u32, 1, 1); //TODO size
        }
        lhs.env().queue.submit(Some(encoder.finish()));

        Tensor::create_from_raw(lhs.env(), lhs.dimensions().to_vec(), Arc::new(result_buf))
    }

    fn grad<'env>(&self, lhs: &Tensor<'env, WgpuEnv>, lhs_grad: &Option<Tensor<'env, WgpuEnv>>, rhs: &Tensor<'env, WgpuEnv>, rhs_grad: &Option<Tensor<'env, WgpuEnv>>) -> Option<Tensor<'env, WgpuEnv>> {
        todo!()
    }
}

#[cfg(test)]
mod test {
    use rstest::rstest;
    use crate::operations::binop_plus::BinOpPlus;
    use crate::tensor_env::TensorEnv;
    use crate::test_utils::tensor_factories::tensor_from_spec;
    use crate::tracker::BinaryTensorOp;
    use crate::with_all_envs;

    #[rstest]
    #[case("1.0", "2.0", "3.0")]
    #[case("R:[1, 2, 3]", "R:[4, 5, 6]", "R:[5, 7, 9]")]
    fn test_add(#[case] a: &str, #[case] b: &str, #[case] expected: &str) {
        with_all_envs!(env => {
            let a = tensor_from_spec(a, &env);
            let b = tensor_from_spec(b, &env);
            let c = BinOpPlus{}.calc(&a, &b);

            c.assert_pretty_much_equal_to(&tensor_from_spec(expected, &env));
        })
    }
}

















