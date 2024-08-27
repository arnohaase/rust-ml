use blas::sscal;

use crate::operations::binop_plus::BinOpPlus;
use crate::operations::calc_utils::chunk_wise_bin_op;
use crate::tensor::Tensor;
use crate::tensor_env::{BlasEnv, TensorEnv};
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
    return regular.env().create_tensor(regular.dimensions().into(), result);
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


#[cfg(test)]
mod test {
    use rstest::rstest;

    use crate::operations::binop_mult::raw_mult_blas;
    use crate::tensor_env::BlasEnv;
    use crate::test_utils::tensor_factories::tensor_from_spec;

    #[rstest]
    #[case("C:[1,2,3]", "C-P:[[3,4][5,6][7,8]]", "C-P:[[3,4][10,12][21,24]]")]
    fn test_calc(#[case] lhs_spec: &str, #[case] rhs_spec: &str, #[case] expected_spec: &str) {
        let env = BlasEnv{};

        let lhs = tensor_from_spec(lhs_spec, &env);
        let rhs = tensor_from_spec(rhs_spec, &env);
        let expected = tensor_from_spec(expected_spec, &env);

        println!("{:?} * {:?} ?= {:?}", lhs, rhs, expected);
        raw_mult_blas(&lhs, &rhs).assert_pretty_much_equal_to(&expected);
        raw_mult_blas(&rhs, &lhs).assert_pretty_much_equal_to(&expected);
    }
}

