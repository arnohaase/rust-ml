use crate::dimension::{Dimensions, MatchDimensionsResult};
use crate::tensor::Tensor;
use crate::tensor_env::{BlasEnv, TensorEnv};

pub fn chunk_wise_bin_op<'env>(
    lhs: &Tensor<'env, BlasEnv>,
    rhs: &Tensor<'env, BlasEnv>,
    is_commutative: bool,
    chunk_op: impl Fn(usize, &[f32], usize, &mut [f32], usize),
) -> Tensor<'env, BlasEnv> {
    let lhs_buf = lhs.buf().read().unwrap();
    let rhs_buf = rhs.buf().read().unwrap();

    let lhs_dim = lhs.dimensions();
    let rhs_dim = rhs.dimensions();

    if is_commutative && rhs_dim.num_dims() > lhs_dim.num_dims() {
        // implementations (especially BLAS based) tend to be optimized for LHS > RHS
        _chunk_wise_bin_op(&rhs_buf, &lhs_buf, rhs_dim, lhs_dim, chunk_op, lhs.env())
    }
    else {
        _chunk_wise_bin_op(&lhs_buf, &rhs_buf, lhs_dim, rhs_dim, chunk_op, lhs.env())
    }
}

fn _chunk_wise_bin_op<'env>(
    lhs_buf: &[f32],
    rhs_buf: &[f32],
    lhs_dim: &Dimensions,
    rhs_dim: &Dimensions,
    chunk_op: impl Fn(usize, &[f32], usize, &mut [f32], usize),
    env: &'env BlasEnv,
) -> Tensor<'env, BlasEnv> {
    let mut result_buf: Vec<f32>;
    let result_dim: Dimensions;
    match lhs_dim.match_with_other(rhs_dim) {
        MatchDimensionsResult::Mismatch =>
            todo!("dimension mismatch"),
        MatchDimensionsResult::Equal => {
            result_buf = Vec::with_capacity(lhs_buf.len());
            result_buf.extend_from_slice(&lhs_buf);
            result_dim = lhs_dim.clone();
            chunk_op(rhs_buf.len(), &rhs_buf, 1, &mut result_buf, 1);
        }
        MatchDimensionsResult::LeftContainsRight { num_wrapper_dims, num_nested_dims } => {
            result_buf = Vec::with_capacity(lhs_buf.len());
            result_dim = lhs_dim.clone();

            //TODO extract to 'Dimensions' type
            let chunk_size = lhs_dim.size_without_outer(num_wrapper_dims);
            let num_interleaved: usize = lhs_dim.size_inner(num_nested_dims);
            for lhs_chunk in lhs_buf.chunks(chunk_size) {
                let chunk_offs = result_buf.len();
                result_buf.extend_from_slice(lhs_chunk);

                for interleave_offset in 0..num_interleaved {
                    chunk_op(rhs_buf.len(), &rhs_buf, 1, &mut result_buf[chunk_offs + interleave_offset..], num_interleaved);
                }
            }
        }
        MatchDimensionsResult::RightContainsLeft { num_wrapper_dims, num_nested_dims } => {
            result_buf = Vec::with_capacity(rhs_buf.len());
            result_dim = rhs_dim.clone();

            let nested_size = rhs_dim.size_inner(num_nested_dims);
            let chunk_size = rhs_dim.size_without_outer(num_wrapper_dims);

            for rhs_chunk in rhs_buf.chunks(chunk_size) {
                let chunk_offs = result_buf.len();
                for lhs_el in lhs_buf.iter() {
                    for _ in 0..nested_size {
                        result_buf.push(*lhs_el);
                    }
                }
                chunk_op(chunk_size, rhs_chunk, 1, &mut result_buf[chunk_offs..], 1);
            }
        }
    }
    env.create_tensor(result_dim.into(), result_buf)
}
