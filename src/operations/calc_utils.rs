use std::cmp::Ordering;
use crate::tensor::{Dimension, Tensor};
use crate::tensor_env::{BlasEnv, TensorEnv};

pub enum FitDimensionsResult {
    Equal,
    Mismatch,
    LeftContainsRight { num_wrapper_dims: usize, num_nested_dims: usize, },
    RightContainsLeft { num_wrapper_dims: usize, num_nested_dims: usize, },
}

/// Compares two tensors' dimensions, checking if one of the tensors contains parts with the
///  other's dimensions
pub fn fit_dimensions(lhs_dim: &[Dimension], rhs_dim: &[Dimension]) -> FitDimensionsResult {
    //TODO ambiguity?
    //TODO for nested_dims in 0..lhs.len() - rhs.len()

    match lhs_dim.len().cmp(&rhs_dim.len()) {
        Ordering::Equal =>
            if lhs_dim == rhs_dim { FitDimensionsResult::Equal } else { FitDimensionsResult::Mismatch }
        Ordering::Less =>
            check_dims_contained(rhs_dim, lhs_dim,
                                 |num_wrapper_dims, num_nested_dims| FitDimensionsResult::RightContainsLeft { num_wrapper_dims, num_nested_dims, }),
        Ordering::Greater =>
            check_dims_contained(lhs_dim, rhs_dim,
                                 |num_wrapper_dims, num_nested_dims| FitDimensionsResult::LeftContainsRight { num_wrapper_dims, num_nested_dims, }),
    }
}

fn check_dims_contained(
    longer: &[Dimension],
    shorter: &[Dimension],
    factory: impl FnOnce(usize, usize) -> FitDimensionsResult,
) -> FitDimensionsResult {
    for num_nested_dims in 0..= longer.len() - shorter.len() {
        if longer[..longer.len()- num_nested_dims].ends_with(shorter) {
            return factory(longer.len() - shorter.len() - num_nested_dims, num_nested_dims);
        }
    }
    FitDimensionsResult::Mismatch
}

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

    if is_commutative && rhs_dim.len() > lhs_dim.len() {
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
    lhs_dim: &[Dimension],
    rhs_dim: &[Dimension],
    chunk_op: impl Fn(usize, &[f32], usize, &mut [f32], usize),
    env: &'env BlasEnv,
) -> Tensor<'env, BlasEnv> {
    let mut result_buf: Vec<f32>;
    let result_dim: Vec<Dimension>;
    match fit_dimensions(lhs_dim, rhs_dim) {
        FitDimensionsResult::Mismatch =>
            todo!("dimension mismatch"),
        FitDimensionsResult::Equal => {
            result_buf = Vec::with_capacity(lhs_buf.len());
            result_buf.extend_from_slice(&lhs_buf);
            result_dim = lhs_dim.to_vec();
            chunk_op(rhs_buf.len(), &rhs_buf, 1, &mut result_buf, 1);
        }
        FitDimensionsResult::LeftContainsRight { num_wrapper_dims, num_nested_dims } => {
            result_buf = Vec::with_capacity(lhs_buf.len());
            result_dim = lhs_dim.to_vec();

            //TODO extract to 'Dimensions' type
            let chunk_size = lhs_dim[num_wrapper_dims..].iter().map(|d| d.len).product(); // empty --> 1
            let num_interleaved: usize = lhs_dim[lhs_dim.len() - num_nested_dims..].iter().map(|d| d.len).product();
            for lhs_chunk in lhs_buf.chunks(chunk_size) {
                let chunk_offs = result_buf.len();
                result_buf.extend_from_slice(lhs_chunk);

                for interleave_offset in 0..num_interleaved {
                    chunk_op(rhs_buf.len(), &rhs_buf, 1, &mut result_buf[chunk_offs + interleave_offset..], num_interleaved);
                }
            }
        }
        FitDimensionsResult::RightContainsLeft { num_wrapper_dims, num_nested_dims } => {
            result_buf = Vec::with_capacity(rhs_buf.len());
            result_dim = rhs_dim.into();

            let nested_size: usize = rhs_dim[num_wrapper_dims + lhs_dim.len()..] //TODO extract to 'Dimensions' type
                .iter()
                .map(|d| d.len)
                .product();

            let chunk_size = rhs_dim[num_wrapper_dims..].iter().map(|d| d.len).product(); // empty --> 1
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
    env.create_tensor(result_dim, result_buf)
}


#[cfg(test)]
mod test {
    fn test_fit_dimensions() {
        todo!()
    }
}