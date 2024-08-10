use crate::tensor::Tensor;
use crate::tensor_env::{BlasEnv, TensorEnv};
use crate::tracker::BinaryTensorOp;

/// treats the first argument (which must be a vector) as scalar coefficients in a polynomial,
///  applying that polynomial element-wise to the second argument.
///
/// The index of a coefficient in the first argument is the power it corresponds to, i.e. the first
///  element is a constant, the second is the coefficient for linear etc.
#[derive(Debug)]
pub struct BinOpPolynomial{}
impl BinaryTensorOp<BlasEnv> for BinOpPolynomial {
    fn calc<'env>(&self, lhs: &Tensor<'env, BlasEnv>, rhs: &Tensor<'env, BlasEnv>) -> Tensor<'env, BlasEnv> {
        assert!(lhs.is_vector()); //TODO collection of polynomials
        //TODO assert that lhs has kind 'polynomial'

        let lhs_buf = lhs.buf().read().unwrap();
        let rhs_buf = rhs.buf().read().unwrap();

        let mut result_buf = Vec::with_capacity(rhs_buf.len());

        //TODO specialize for different lengths of lhs

        for &x in rhs_buf.iter() {
            let mut new_x = lhs_buf[0];

            let mut x_pow = x;
            for &coeff in &lhs_buf[1..] {
                new_x += coeff * x_pow;
                x_pow *= x;
            }
            result_buf.push(new_x);
        }
        lhs.env().create_tensor(rhs.dimensions().to_vec(), result_buf)
    }

    fn grad<'env>(&self, lhs: &Tensor<'env, BlasEnv>, lhs_grad: &Option<Tensor<'env, BlasEnv>>, rhs: &Tensor<'env, BlasEnv>, rhs_grad: &Option<Tensor<'env, BlasEnv>>) -> Option<Tensor<'env, BlasEnv>> {
        match (lhs_grad.as_ref(), rhs_grad.as_ref()) {
            (None, None) => None,
            (Some(lhs_grad), None) => {
                if lhs_grad.is_scalar() {
                    let mut grad = vec![];
                    let poly_dim = lhs.dimensions()[0];
                    let lhs_grad = lhs_grad.buf().read().unwrap()[0];

                    for x in rhs.buf().read().unwrap().iter() {
                        let mut x_pow = 1.0;
                        for _ in 0..poly_dim.len {
                            grad.push(lhs_grad * x_pow);
                            x_pow *= x;
                        }
                    }
                    let mut grad_dimensions = rhs.dimensions().to_vec();
                    grad_dimensions.push(poly_dim);
                    Some(lhs.env().create_tensor(grad_dimensions, grad))
                }
                else {
                    todo!()
                }
            },
            (None, Some(rhs_grad)) => {
                todo!()
            },
            (Some(lhs_grad), Some(rhs_grad)) => todo!(),
        }
    }
}

#[cfg(test)]
mod test {
    use rstest::rstest;

    use crate::operations::binop_polynomial::BinOpPolynomial;
    use crate::tensor::{DimensionKind, Tensor};
    use crate::tensor_env::{BlasEnv, TensorEnv};
    use crate::tracker::{BinaryTensorOp, ExecutionTracker, RegularExecutionTracker, TrackerExpression};

    #[rstest]
    #[case::constant(vec![5.0], 5.0)]
    #[case::linear(vec![1.0, 3.0], 7.0)]
    #[case::quadratic(vec![1.0, 3.0, 4.0], 23.0)]
    fn test_calc(#[case] poly: Vec<f64>, #[case] expected_calc_result: f64) {
        let env = BlasEnv {};
        let x = 2.0;
        let actual = BinOpPolynomial{}.calc(&env.vector(poly, DimensionKind::Polynomial), &env.scalar(x));
        actual.assert_pretty_much_equal_to(&env.scalar(expected_calc_result));
    }

    #[test]
    fn test_grad() {
        let env = BlasEnv{};

        let poly_coefficients = env.vector(vec!(1.0, 2.0, 3.0, 4.0), DimensionKind::Polynomial);
        let x = env.scalar(2.0);

        let mut tracker = RegularExecutionTracker::new();
        let calc_result = tracker.calc(TrackerExpression::Binary(poly_coefficients.clone(), x.clone(), Box::new(BinOpPolynomial {})));
        calc_result.assert_pretty_much_equal_to(&env.scalar(1.0 + 4.0 + 12.0 + 32.0));

        let grad = tracker.grad(&calc_result, &poly_coefficients).unwrap();
        grad.assert_pretty_much_equal_to(&env.vector(vec![1.0, 2.0, 4.0, 8.0], DimensionKind::Polynomial));
    }
}
