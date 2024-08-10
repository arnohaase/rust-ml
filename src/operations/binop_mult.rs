use blas::dscal;

use crate::operations::binop_plus::BinOpPlus;
use crate::operations::calc_utils::chunk_wise_bin_op;
use crate::tensor::{BlasEnv, Tensor, TensorEnv};
use crate::tracker::BinaryTensorOp;

#[derive(Debug)]
pub struct BinOpMult {}
impl BinOpMult {
    pub fn new() -> BinOpMult {
        BinOpMult{}
    }

    pub fn raw_mult<'env>(lhs: &Tensor<'env, BlasEnv>, rhs: &Tensor<'env, BlasEnv>) -> Tensor<'env, BlasEnv> {
        // if lhs.is_one() {
        //     return rhs.clone_with_new_id();
        // }
        // if rhs.is_one() {
        //     return lhs.clone_with_new_id();
        // }

        if lhs.is_scalar() {
            return Self::raw_mult_scalar(lhs, rhs);
        }
        if rhs.is_scalar() {
            return Self::raw_mult_scalar(rhs, lhs);
        }

        chunk_wise_bin_op(lhs, rhs, true, Self::raw_mult_chunk)
    }

    fn raw_mult_scalar<'env>(scalar: &Tensor<'env, BlasEnv>, regular: &Tensor<'env, BlasEnv>) -> Tensor<'env, BlasEnv> {
        let scalar = scalar.buf().read().unwrap()[0];
        let mut result = regular.buf().read().unwrap().to_vec();
        unsafe {
            dscal(result.len() as i32, scalar, result.as_mut_slice(), 1);
        }
        return regular.env().create_tensor(regular.dimensions().into(), result);
    }

    fn raw_mult_chunk(n: usize, rhs: &[f64], inc_rhs: usize, lhs: &mut[f64], inc_lhs: usize) {
        let mut offs_lhs = 0;
        let mut offs_rhs = 0;

        for _ in 0..n {
            lhs[offs_lhs] *= rhs[offs_rhs];
            offs_lhs += inc_lhs;
            offs_rhs += inc_rhs;
        }
    }
}
impl BinaryTensorOp<BlasEnv> for BinOpMult {
    fn calc<'env>(&self, lhs: &Tensor<'env, BlasEnv>, rhs: &Tensor<'env, BlasEnv>) -> Tensor<'env, BlasEnv> {
        Self::raw_mult(lhs, rhs)
    }

    fn grad<'env>(&self, lhs: &Tensor<'env, BlasEnv>, lhs_grad: &Option<Tensor<'env, BlasEnv>>, rhs: &Tensor<'env, BlasEnv>, rhs_grad: &Option<Tensor<'env, BlasEnv>>) -> Option<Tensor<'env, BlasEnv>> {
        match (lhs_grad, rhs_grad) {
            (None, None) => None,
            (Some(lhs_grad), None) => Some(Self::raw_mult(lhs_grad, rhs)),
            (None, Some(rhs_grad)) => Some(Self::raw_mult(lhs, rhs_grad)),
            (Some(lhs_grad), Some(rhs_grad)) => {
                let a = Self::raw_mult(lhs_grad, rhs);
                let b = Self::raw_mult(lhs, rhs_grad);

                Some(BinOpPlus::raw_plus(&a, &b))
            },
        }
    }
}

#[cfg(test)]
mod test {
    use rstest::rstest;

    use crate::operations::binop_mult::BinOpMult;
    use crate::tensor::{BlasEnv, Dimension, DimensionKind, Tensor};
    use crate::test_utils::tensor_factories::tensor_from_spec;

    #[rstest]
    #[case("C:[1,2,3]", "C-P:[[3,4][5,6][7,8]]", "C-P:[[3,4][10,12][21,24]]")]
    fn test_calc(#[case] lhs_spec: &str, #[case] rhs_spec: &str, #[case] expected_spec: &str) {
        let env = BlasEnv{};

        let lhs = tensor_from_spec(lhs_spec, &env);
        let rhs = tensor_from_spec(rhs_spec, &env);
        let expected = tensor_from_spec(expected_spec, &env);

        println!("{:?} * {:?} ?= {:?}", lhs, rhs, expected);
        BinOpMult::raw_mult(&lhs, &rhs).assert_pretty_much_equal_to(&expected);
        BinOpMult::raw_mult(&rhs, &lhs).assert_pretty_much_equal_to(&expected);
    }
}

