use blas::sscal;

use crate::operations::binop_plus::BinOpPlus;
use crate::operations::calc_utils_blas::chunk_wise_bin_op;
use crate::operations::calc_utils_wgpu::call_shader_binop;
use crate::tensor::Tensor;
use crate::tensor_env::{BlasEnv, TensorEnv, WgpuEnv};
use crate::tracker::BinaryTensorOp;

#[derive(Debug)]
pub struct BinOpMult {}
impl BinOpMult {
    pub fn new() -> BinOpMult {
        BinOpMult{}
    }
}


impl BinaryTensorOp<BlasEnv> for BinOpMult {
    fn calc<'env>(&self, lhs: &Tensor<'env, BlasEnv>, rhs: &Tensor<'env, BlasEnv>) -> Tensor<'env, BlasEnv> {
        raw_mult_blas(lhs, rhs)
    }

    fn grad<'env>(&self, lhs: &Tensor<'env, BlasEnv>, lhs_grad: &Option<Tensor<'env, BlasEnv>>, rhs: &Tensor<'env, BlasEnv>, rhs_grad: &Option<Tensor<'env, BlasEnv>>) -> Option<Tensor<'env, BlasEnv>> {
        match (lhs_grad, rhs_grad) {
            (None, None) => None,
            (Some(lhs_grad), None) => Some(raw_mult_blas(lhs_grad, rhs)),
            (None, Some(rhs_grad)) => Some(raw_mult_blas(lhs, rhs_grad)),
            (Some(lhs_grad), Some(rhs_grad)) => {
                let a = raw_mult_blas(lhs_grad, rhs);
                let b = raw_mult_blas(lhs, rhs_grad);

                Some(BinOpPlus::raw_plus(&a, &b))
            },
        }
    }
}

pub fn raw_mult_blas<'env>(lhs: &Tensor<'env, BlasEnv>, rhs: &Tensor<'env, BlasEnv>) -> Tensor<'env, BlasEnv> {
    // if lhs.is_one() {
    //     return rhs.clone_with_new_id();
    // }
    // if rhs.is_one() {
    //     return lhs.clone_with_new_id();
    // }

    if lhs.is_scalar() {
        return raw_mult_scalar_blas(lhs, rhs);
    }
    if rhs.is_scalar() {
        return raw_mult_scalar_blas(rhs, lhs);
    }

    chunk_wise_bin_op(lhs, rhs, true, raw_mult_chunk_blas)
}

fn raw_mult_scalar_blas<'env>(scalar: &Tensor<'env, BlasEnv>, regular: &Tensor<'env, BlasEnv>) -> Tensor<'env, BlasEnv> {
    let scalar = scalar.buf().read().unwrap()[0];
    let mut result = regular.buf().read().unwrap().to_vec();
    unsafe {
        sscal(result.len() as i32, scalar, result.as_mut_slice(), 1);
    }
    return regular.env().create_tensor(regular.dimensions().clone(), result);
}

fn raw_mult_chunk_blas(n: usize, rhs: &[f32], inc_rhs: usize, lhs: &mut[f32], inc_lhs: usize) {
    let mut offs_lhs = 0;
    let mut offs_rhs = 0;

    for _ in 0..n {
        lhs[offs_lhs] *= rhs[offs_rhs];
        offs_lhs += inc_lhs;
        offs_rhs += inc_rhs;
    }
}


impl BinaryTensorOp<WgpuEnv> for BinOpMult {
    fn calc<'env>(&self, lhs: &Tensor<'env, WgpuEnv>, rhs: &Tensor<'env, WgpuEnv>) -> Tensor<'env, WgpuEnv> {
        call_shader_binop(lhs, rhs, "*", include_str!("binop_mult.wgsl"), None)
    }

    fn grad<'env>(&self, lhs: &Tensor<'env, WgpuEnv>, lhs_grad: &Option<Tensor<'env, WgpuEnv>>, rhs: &Tensor<'env, WgpuEnv>, rhs_grad: &Option<Tensor<'env, WgpuEnv>>) -> Option<Tensor<'env, WgpuEnv>> {
        match (lhs_grad, rhs_grad) {
            (None, None) => None,
            (Some(lhs_grad), None) => Some(BinOpMult{}.calc(lhs_grad, rhs)),
            (None, Some(rhs_grad)) => Some(BinOpMult{}.calc(lhs, rhs_grad)),
            (Some(lhs_grad), Some(rhs_grad)) => Some(BinOpPlus{}.calc(
                &BinOpMult{}.calc(lhs_grad, rhs),
                &BinOpMult{}.calc(lhs, rhs_grad),
            ))
        }
    }
}

#[cfg(test)]
mod test {
    use rstest::rstest;

    use crate::operations::binop_mult::BinOpMult;
    use crate::test_utils::tensor_factories::tensor_from_spec;
    use crate::tracker::BinaryTensorOp;
    use crate::with_all_envs;

    #[rstest]
    #[case::scalar("2.0", "3.0", "6.0")]
    #[case::simple_vec("R:[1, 2, 3]", "R:[4, 5, 6]", "R:[4, 10, 18]")]
    #[case::nested_left("R-P:[[1,2][3,4]]", "R:[5,6]", "R-P:[[5, 10][18, 24]]")]
    #[case::nested_right("R:[5,6]", "R-P:[[1,2][3,4]]", "R-P:[[5, 10][18, 24]]")]
    #[case::collection_left("C-R:[[1,2,3][4,5,6]]", "R:[2,3,4]", "C-R:[[2,6,12][8,15,24]]")]
    #[case::collection_right("R:[2,3,4]", "C-R:[[1,2,3][4,5,6]]", "C-R:[[2,6,12][8,15,24]]")]
    #[case::both_left("C-R-P:[[[1,2,3]][[4,5,6]]]", "R:[.5]", "C-R-P:[[[0.5,1,1.5]][[2,2.5,3]]]")]
    #[case::both_right("R:[.5]", "C-R-P:[[[1,2,3]][[4,5,6]]]", "C-R-P:[[[0.5,1,1.5]][[2,2.5,3]]]")]
    fn test_mult(#[case] a: &str, #[case] b: &str, #[case] expected: &str) {
        with_all_envs!(env => {
            let a = tensor_from_spec(a, &env);
            let b = tensor_from_spec(b, &env);
            let c = BinOpMult{}.calc(&a, &b);

            c.assert_pretty_much_equal_to(&tensor_from_spec(expected, &env));
        })
    }

    //TODO test grad
}
