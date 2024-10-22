use triomphe::Arc;
use wgpu::BufferAddress;

use crate::dimension::{Dimensions, MatchDimensionsResult};
use crate::tensor::Tensor;
use crate::tensor_env::WgpuEnv;

pub fn call_shader_binop<'env>(
    lhs: &Tensor<'env, WgpuEnv>,
    rhs: &Tensor<'env, WgpuEnv>,
    shader_id: &str,
    shader_template: &str,
    shader_template_r: Option<&str>,
) -> Tensor<'env, WgpuEnv> {
    match lhs.dimensions().match_with_other(rhs.dimensions()) {
        MatchDimensionsResult::Mismatch => todo!("dimension mismatch"),
        MatchDimensionsResult::Equal => apply_binop_shader(lhs, rhs, lhs, 1, lhs.buf().size() as usize, 1, shader_id, shader_template),
        MatchDimensionsResult::LeftContainsRight { num_wrapper_dims, num_nested_dims } => {
            let num_chunks = lhs.dimensions().size_outer(num_wrapper_dims);
            let chunk_size = lhs.dimensions().size_without_outer(num_wrapper_dims);
            let interleaved_size: usize = lhs.dimensions().size_inner(num_nested_dims);
            apply_binop_shader(lhs, rhs, lhs, num_chunks, chunk_size, interleaved_size, shader_id, shader_template)
        },
        MatchDimensionsResult::RightContainsLeft { num_wrapper_dims, num_nested_dims } => {
            let num_chunks = rhs.dimensions().size_outer(num_wrapper_dims);
            let chunk_size = rhs.dimensions().size_without_outer(num_wrapper_dims);
            let interleaved_size: usize = rhs.dimensions().size_inner(num_nested_dims);

            if let Some(shader_template) = shader_template_r {
                let shader_id = format!("{shader_id}_r");
                apply_binop_shader(lhs, rhs, rhs, num_chunks, chunk_size, interleaved_size, &shader_id, shader_template)
            }
            else {
                apply_binop_shader(rhs, lhs, rhs, num_chunks, chunk_size, interleaved_size, shader_id, shader_template)
            }
        }
    }
}

fn apply_binop_shader<'env>(
    lhs: &Tensor<'env, WgpuEnv>,
    rhs: &Tensor<'env, WgpuEnv>,
    leading_tensor: &Tensor<'env, WgpuEnv>,
    num_chunks: usize,
    chunk_size: usize,
    interleave_size: usize,
    shader_id: &str,
    shader_template: &str,
) -> Tensor<'env, WgpuEnv> {
    let shader_id = format!("{shader_id}:{interleave_size}:{chunk_size}");

    let shader = lhs.env().shader(
        &shader_id,
        || shader_template
            .replace("{n}", &interleave_size.to_string())
            .replace("{chunk_size}", &chunk_size.to_string())
    );

    let result_buf = lhs.env().create_storage_buffer(leading_tensor.buf().size());

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
    Tensor::create_from_raw(lhs.env(), leading_tensor.dimensions().clone(), Arc::new(result_buf))
}

pub fn apply_unop_shader<'env>(
    tensor: &Tensor<'env, WgpuEnv>,
    result_dimensions: Dimensions,
    shader_id: &str,
    shader_template: &str,
) -> Tensor<'env, WgpuEnv> {
    let shader = tensor.env().shader(
        &shader_id,
        || shader_template
            .replace("{workgroup_size}", "64") //TODO workgroup size
    );

    let result_buf = tensor.env().create_storage_buffer((result_dimensions.size() * size_of::<f32>()) as BufferAddress);

    let bind_group = shader.bind_group(tensor.env(), vec![
        tensor.buf().as_entire_binding(),
        result_buf.as_entire_binding(),
    ]);

    let mut encoder = tensor.env().device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(&shader.id) });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(&shader.id),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&shader.compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(1, (result_dimensions.size() / 64) as u32, 1); //TODO workgroup size
    }

    tensor.env().queue.submit(Some(encoder.finish()));
    Tensor::create_from_raw(tensor.env(), result_dimensions, Arc::new(result_buf))
}
