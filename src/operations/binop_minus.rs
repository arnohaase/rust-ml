use blas::saxpy;
use crate::operations::binop_plus::BinOpPlus;

use crate::operations::calc_utils_blas::chunk_wise_bin_op;
use crate::operations::calc_utils_wgpu::call_shader_binop;
use crate::operations::unop_minus::UnOpMinus;
use crate::tensor::Tensor;
use crate::tensor_env::{BlasEnv, WgpuEnv};
use crate::tracker::BinaryTensorOp;

pub fn raw_minus<'env>(lhs: &Tensor<'env, BlasEnv>, rhs: &Tensor<'env, BlasEnv>) -> Tensor<'env, BlasEnv> {
    BinOpMinus{}.calc(lhs, rhs)
}


#[derive(Debug)]
pub struct BinOpMinus {}
impl BinOpMinus {
    pub fn new() -> BinOpMinus {
        BinOpMinus{}
    }

    pub fn raw_minus<'env>(lhs: &Tensor<'env, BlasEnv>, rhs: &Tensor<'env, BlasEnv>) -> Tensor<'env, BlasEnv> {
        // if rhs.is_zero() {
        //     return lhs.clone_with_new_id();
        // }
        // if lhs.is_zero() {
        //     let mut result = rhs.buf().read().unwrap().clone();
        //     unsafe {
        //         dscal(result.len() as i32, -1.0, result.as_mut_slice(), 1);
        //     }
        //     return Tensor::from_raw(lhs.dimensions().into(), result);
        // }
        chunk_wise_bin_op(lhs, rhs, false, Self::raw_minus_chunk)
    }

    fn raw_minus_chunk(n: usize, rhs: &[f32], inc_rhs: usize, lhs: &mut[f32], inc_lhs: usize) {
        unsafe {
            saxpy(n as i32, -1.0, rhs, inc_rhs as i32, lhs, inc_lhs as i32);
        }
    }
}
impl BinaryTensorOp<BlasEnv> for BinOpMinus {
    fn calc<'env>(&self, lhs: &Tensor<'env, BlasEnv>, rhs: &Tensor<'env, BlasEnv>) -> Tensor<'env, BlasEnv> {
        Self::raw_minus(lhs, rhs)
    }

    fn grad<'env>(&self, _lhs: &Tensor<'env, BlasEnv>, lhs_grad: &Option<Tensor<'env, BlasEnv>>, _rhs: &Tensor<'env, BlasEnv>, rhs_grad: &Option<Tensor<'env, BlasEnv>>) -> Option<Tensor<'env, BlasEnv>> {
        match (lhs_grad, rhs_grad) {
            (None, None) => None,
            (Some(lhs_grad), None) => Some(lhs_grad.clone()),
            (None, Some(rhs_grad)) => Some(UnOpMinus::raw_minus(rhs_grad)),
            (Some(lhs_grad), Some(rhs_grad)) => Some(raw_minus(lhs_grad, rhs_grad)),
        }
    }
}

impl BinaryTensorOp<WgpuEnv> for BinOpMinus {
    fn calc<'env>(&self, lhs: &Tensor<'env, WgpuEnv>, rhs: &Tensor<'env, WgpuEnv>) -> Tensor<'env, WgpuEnv> {
        call_shader_binop(lhs, rhs, "-", include_str!("binop_minus.wgsl"), Some(include_str!("binop_minus_r.wgsl")))
    }

    fn grad<'env>(&self, _lhs: &Tensor<'env, WgpuEnv>, lhs_grad: &Option<Tensor<'env, WgpuEnv>>, _rhs: &Tensor<'env, WgpuEnv>, rhs_grad: &Option<Tensor<'env, WgpuEnv>>) -> Option<Tensor<'env, WgpuEnv>> {
        match (lhs_grad, rhs_grad) {
            (None, None) => None,
            (Some(lhs_grad), None) => Some(lhs_grad.clone_with_new_id()),
            (None, Some(rhs_grad)) => todo!("Some(-rhs_grad.clone_with_new_id())"),
            (Some(lhs_grad), Some(rhs_grad)) => Some(self.calc(lhs_grad, rhs_grad)),
        }
    }
}

#[cfg(test)]
mod test {
    use rstest::rstest;
    use crate::operations::binop_minus::BinOpMinus;
    use crate::test_utils::tensor_factories::tensor_from_spec;
    use crate::tracker::BinaryTensorOp;
    use crate::with_all_envs;

    #[rstest]
    #[case::scalar("1.0", "2.0", "-1.0")]
    #[case::simple_vec("R:[1, 2, 3]", "R:[4, 5, 6]", "R:[-3, -3, -3]")]
    #[case::nested_left("R-P:[[1,2][3,4]]", "R:[5,6]", "R-P:[[-4,-3][-3,-2]]")]
    #[case::nested_right("R:[5,6]", "R-P:[[1,2][3,4]]", "R-P:[[4,3][3,2]]")]
    #[case::collection_left("C-R:[[1,2,3][4,5,6]]", "R:[2,3,4]", "C-R:[[-1,-1,-1][2,2,2]]")]
    #[case::collection_right("R:[2,3,4]", "C-R:[[1,2,3][4,5,6]]", "C-R:[[1,1,1][-2,-2,-2]]")]
    #[case::both_left("C-R-P:[[[1,2,3]][[4,5,6]]]", "R:[.5]", "C-R-P:[[[0.5,1.5,2.5]][[3.5,4.5,5.5]]]")]
    #[case::both_right("R:[.5]", "C-R-P:[[[1,2,3]][[4,5,6]]]", "C-R-P:[[[-0.5,-1.5,-2.5]][[-3.5,-4.5,-5.5]]]")]
    fn test_minus(#[case] a: &str, #[case] b: &str, #[case] expected: &str) {
        with_all_envs!(env => {
            let a = tensor_from_spec(a, &env);
            let b = tensor_from_spec(b, &env);
            let c = BinOpMinus{}.calc(&a, &b);

            c.assert_pretty_much_equal_to(&tensor_from_spec(expected, &env));
        })
    }

    //TODO test grad
}
