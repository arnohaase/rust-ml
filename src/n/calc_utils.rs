use std::cmp::Ordering;
use crate::n::tensor::{Dimension, Tensor};

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
    for num_nested_dims in 1..= longer.len() - shorter.len() {
        if longer[..longer.len()- num_nested_dims].ends_with(shorter) {
            return factory(longer.len() - shorter.len() - num_nested_dims, num_nested_dims);
        }
    }
    FitDimensionsResult::Mismatch
}

pub fn chunk_wise_bin_op(lhs: &Tensor, rhs: &Tensor, chunk_op: impl Fn(&[f64], &[f64], &mut Vec<f64>)) -> Tensor {
    let lhs_buf = lhs.buf().read().unwrap();
    let rhs_buf = rhs.buf().read().unwrap();
    let mut result_buf: Vec<f64>;
    let result_dim: Vec<usize>;
    match fit_dimensions(lhs.dimensions(), rhs.dimensions()) {
        FitDimensionsResult::Mismatch =>
            todo!("dimension mismatch"),
        FitDimensionsResult::Equal => {
            result_buf = Vec::with_capacity(lhs_buf.len());
            result_dim = lhs.dimensions().into();
            chunk_op(&lhs_buf, &rhs_buf, &mut result_buf);
        }
        FitDimensionsResult::LeftContainsRight { num_wrapper_dims, num_nested_dims } => {
            result_buf = Vec::with_capacity(lhs_buf.len());
            result_dim = lhs.dimensions().into();
            for lhs_chunk in lhs_buf.chunks(chunk_size) {
                chunk_op(lhs_chunk, &rhs_buf, &mut result_buf);
            }
        }
        FitDimensionsResult::RightContainsLeft { chunk_size } => {
            result_buf = Vec::with_capacity(rhs_buf.len());
            result_dim = rhs.dimensions().into();
            for rhs_chunk in rhs_buf.chunks(chunk_size) {
                chunk_op(&lhs_buf, rhs_chunk, &mut result_buf);
            }
        }
    }
    Tensor::from_raw(result_dim, result_buf)
}


#[cfg(test)]
mod test {
    fn test_fit_dimensions() {
        todo!()
    }
}