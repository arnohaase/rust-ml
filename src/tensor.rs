use std::cmp::max;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::fmt::{Debug, Display, Formatter};
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::atomic::Ordering::Acquire;

use lazy_static::lazy_static;
use rand::random;
use triomphe::Arc;

pub trait Float:
    Copy + Display + Debug +
    PartialOrd +
    Add<Output=Self> + Sub<Output=Self> + Mul<Output=Self> + Div<Output=Self> +
    Neg<Output=Self> +
    From<u16> + From<f64>
{
    fn zero() -> Self;
    fn one() -> Self;
    fn random_0_1() -> Self;

    fn pow_i(self, i: u16) -> Self;

    fn sin(self) -> Self;
}

impl Float for f64 {
    fn zero() -> Self {
        0.0
    }
    fn one() -> f64 {
        1.0
    }

    fn random_0_1() -> f64 {
        random()
    }

    fn pow_i(self, n: u16) -> Self {
        self.powi(n as i32)
    }


    fn sin(self) -> f64 {
        self.sin()
    }
}

lazy_static! {
    static ref TENSOR_ID_COUNTER: AtomicU32 = AtomicU32::new(0);
}

pub struct TensorEnv<F: Float> {
    pd: PhantomData<F>,
}
impl <F: Float> TensorEnv<F> {
    pub fn new() -> TensorEnv<F> {
        TensorEnv {
            pd: Default::default(),
        }
    }

    fn assemble(&self, geometry: Geometry, data: Vec<F>) -> Tensor<F> {
        let id = TENSOR_ID_COUNTER.fetch_add(1, Ordering::SeqCst);
        Tensor {
            env: self,
            id,
            version: Default::default(),
            geometry,
            data: Arc::new(RwLock::new(data)),
        }
    }

    pub fn scalar(&self, value: F) -> Tensor<F> {
        self.assemble(Geometry::scalar(),vec![value])
    }

    pub fn vector(&self, values: Vec<F>) -> Tensor<F> {
        assert!(values.len() > 0); //TODO ?!?!
        self.assemble(Geometry::vector(values.len()), values)
    }

    pub fn random_lin(&self, min: F, max: F, dim: usize) -> Tensor<F> {
        let mut data = Vec::with_capacity(dim);
        for _ in 0..dim {
            data.push(F::random_0_1() * (max-min) + min);
        }
        self.assemble(Geometry::vector(dim), data)
    }
}

#[derive(Clone, PartialEq, Eq)]
pub struct Geometry(Vec<usize>);

impl Debug for Geometry {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self.num_dims() {
            0 => write!(f, "Scalar"),
            1 => write!(f, "Vector{:?}", self.dims()),
            2 => write!(f, "Matrix{:?}", self.dims()),
            _ => write!(f, "Tensor{:?}", self.dims()),
        }
    }
}

impl Geometry {
    pub fn scalar() -> Geometry {
        Geometry (vec![])
    }
    pub fn vector(n: usize) -> Geometry {
        Geometry (vec![n])
    }

    pub fn dims(&self) -> &[usize] {
        &self.0
    }
    pub fn num_dims(&self) -> usize {
        self.0.len()
    }

    pub fn compare_to(&self, other: &Geometry) -> GeometryComparisonResult { //TODO unit test
        if self == other {
            GeometryComparisonResult::Same
        }
        else if self.dims().ends_with(other.dims()) {
            GeometryComparisonResult::SelfContainsOther(self.dims()[0..self.num_dims()-other.num_dims()].into())
        }
        else if other.dims().ends_with(self.dims()) {
            GeometryComparisonResult::OtherContainsSelf(other.dims()[0..other.num_dims()-self.num_dims()].into())
        }
        else {
            GeometryComparisonResult::Unrelated
        }
    }
}

#[derive(Clone, Debug)]
pub enum GeometryComparisonResult {
    Same,
    SelfContainsOther(Vec<usize>),
    OtherContainsSelf(Vec<usize>),
    Unrelated,
}



#[derive(Clone)]
pub struct Tensor<'env, F: Float> {
    //TODO optional name (for debugging)
    env: &'env TensorEnv<F>,
    id: u32,
    /// incremented on each change
    version: Arc<AtomicU32>,
    geometry: Geometry,
    data: Arc<RwLock<Vec<F>>>,
}
impl<'env, F: Float> Debug for Tensor<'env, F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}: {:?}", self.geometry, self.read())
    }
}


impl<'env, F: Float> Tensor<'env, F> {
    pub fn r(&self) -> Tensor<'env, F> {
        self.clone()
    }

    /// Comparing floating point numbers for exact equality does not work much of the time due to
    ///  rounding errors. This is a convenience function that compares two tensors element by
    ///  element, returning true if each of the elements are equal within a heuristically chosen
    ///  EPSILON.
    ///
    /// This method is intended as convenience functionality for unit tests.
    pub fn is_pretty_much_equal_to(&self, other: &Tensor<'env, F>) -> bool {
        let eps:F = 1e-8.into();

        let equivalent_geometry =
            self.geometry == other.geometry
                || self.geometry == Geometry::scalar() && other.geometry == Geometry::vector(1)
                || self.geometry == Geometry::vector(1) && other.geometry == Geometry::scalar()
            ;

        if !equivalent_geometry {
            return false;
        }

        let t1 = self.read();
        let t2 = other.read();
        for i in 0..t1.len() {
            let diff = t1[i] - t2[i];

            if diff > eps || diff < -eps {
                return false;
            }
        }
        true
    }

    pub fn assert_is_pretty_much_equal_to(&self, other: &Tensor<'env, F>) {
        assert!(self.is_pretty_much_equal_to(other), "not equal: {:?} != {:?}", self, &other);
    }

    fn get_version(&self) -> u32 {
        self.version.fetch_add(0, Acquire)
    }

    fn read(&self) -> RwLockReadGuard<'_, Vec<F>> {
        self.data.read().unwrap()
    }
    fn write(&self) -> RwLockWriteGuard<'_, Vec<F>> {
        self.data.write().unwrap()
    }


    /// This handles mismatching geometries by lifting / iterating over outer dimensions
    fn binary_operation(&self, other: &Tensor<'_, F>, f: impl Fn(&[F], &[F], &mut Vec<F>)) -> Tensor<'env, F> {
        let t1 = self.read();
        let t2 = other.read();

        let mut result = Vec::with_capacity(max(t1.len(), t2.len()));

        match self.geometry.compare_to(&other.geometry) {
            GeometryComparisonResult::Same => {
                f(&t1, &t2, &mut result);
                self.env.assemble(self.geometry.clone(), result)
            }
            GeometryComparisonResult::SelfContainsOther(dims) => {
                let num: usize = dims.iter().product();
                for n in 0..num {
                    let offs = n * t2.len();
                    f(&t1[offs..offs+t2.len()], &t2, &mut result);
                }
                self.env.assemble(self.geometry.clone(), result)
            }
            GeometryComparisonResult::OtherContainsSelf(dims) => {
                let num: usize = dims.iter().product();
                for n in 0..num {
                    let offs = n * t1.len();
                    f(&t1, &t2[offs..offs+t1.len()], &mut result);
                }
                self.env.assemble(other.geometry.clone(), result)
            }
            GeometryComparisonResult::Unrelated => {
                panic!("incompatible geometries"); //TODO
            }
        }
    }


    pub fn plus(&self, other: Tensor<'env, F>, tracker: &dyn ExecutionTracker<'env, F>) -> Tensor<'env, F> {
        tracker.calc(TrackerExpression::Two(self.r(), other, Box::new(PlusOp{})))
    }
    fn _plus(&self, rhs: Tensor<'_, F>) -> Tensor<'env, F> {
        self.binary_operation(&rhs, |a, b, result| {
            for i in 0..a.len() {
                result.push(a[i] + b[i]);
            }
        })
    }

    pub fn minus(&self, other: Tensor<'env, F>, tracker: &dyn ExecutionTracker<'env, F>) -> Tensor<'env, F> {
        tracker.calc(TrackerExpression::Two(self.r(), other, Box::new(MinusOp{})))
    }
    fn _minus(&self, rhs: Tensor<'_, F>) -> Tensor<'env, F> {
        self.binary_operation(&rhs, |a, b, result| {
            for i in 0..a.len() {
                result.push(a[i] - b[i]);
            }
        })
    }

    fn sum(&self, tracker: &dyn ExecutionTracker<'env, F>) -> Tensor<'env, F> {
        tracker.calc(TrackerExpression::Single(self.r(), Box::new(SumOp{}))) //TODO convenience API
    }
    fn _sum(&self) -> Tensor<'env, F> {
        let mut result = F::zero();

        for &x in self.read().iter() {
            result = result + x;
        }
        self.env.scalar(result)
    }

    pub fn mult(&self, rhs: Tensor<'env, F>, tracker: &dyn ExecutionTracker<'env, F>) -> Tensor<'env, F> {
        tracker.calc(TrackerExpression::Two(self.r(), rhs, Box::new(MultOp{})))
    }

    fn _mult(&self, rhs: Tensor<'_, F>) -> Tensor<'env, F> {
        self.binary_operation(&rhs, |a, b, result| {
            for i in 0..a.len() {
                result.push(a[i] * b[i]);
            }
        })
    }

    pub fn pow_int(&self, exp: u16, tracker: &dyn ExecutionTracker<'env, F>) -> Tensor<'env, F> {
        tracker.calc(TrackerExpression::Single(self.r(), Box::new(PowInt { exp })))
    }
    pub fn _pow_int(&self, exp: u16) -> Tensor<'env, F> {
        let v1 = self.read();
        let mut data = Vec::with_capacity(v1.len());
        for &x in v1.iter() {
            match exp {
                0 => data.push(F::one()),
                1 => data.push(x),
                2 => data.push(x*x),
                3 => data.push(x*x*x),
                _ => data.push(x.pow_i(exp)), //TODO up to what exponent is multiplying more efficient?
            }
        }
        self.env.assemble(self.geometry.clone(), data)
    }

    //TODO sin()
    pub fn _sin(&self) -> Tensor<'env, F> {
        let v1 = self.read();
        let mut data = Vec::with_capacity(v1.len());
        for &x in v1.iter() {
            data.push(x.sin());
        }
        self.env.assemble(self.geometry.clone(), data)
    }

    pub fn _sub_in_place(&self, other: Tensor<'env, F>) {
        //TODO assert / lift geometry
        let mut data = self.write();
        let mut other = other.read();

        for i in 0..data.len() {
            data[i] = data[i] - other[i];
        }
    }
}


//TODO 'requires_grad' with possible 'ultimate input' variables on construction as an optimization?

pub trait ExecutionTracker<'env, F: Float> {
    fn calc(&self, expr: TrackerExpression<'env, F>) -> Tensor<'env, F>;

    fn grad(&self, calculated: Tensor<'env, F>, for_ultimate_input: Tensor<'env, F>) -> Option<Tensor<'env, F>>;
}

pub struct NoTracker {}
impl <'env, F: Float> ExecutionTracker<'env, F> for NoTracker {
    fn calc(&self, expr: TrackerExpression<'env, F>) -> Tensor<'env, F> {
        expr.calc()
    }

    fn grad(&self, calculated: Tensor<'env, F>, for_ultimate_input: Tensor<'env, F>) -> Option<Tensor<'env, F>> {
        None
    }
}

pub struct RegularExecutionTracker<'env, F: Float> {
    dependencies: RwLock<HashMap<u32, (u32, TrackerExpression<'env, F>)>>,
}
impl <'env, F: Float> RegularExecutionTracker<'env, F> {
    pub fn new() -> RegularExecutionTracker<'env, F> {
        RegularExecutionTracker {
            dependencies: Default::default()
        }
    }
}
impl <'env, F: Float> ExecutionTracker<'env, F> for RegularExecutionTracker<'env, F> {
    fn calc(&self, expr: TrackerExpression<'env, F>) -> Tensor<'env, F> {
        let calculated = expr.calc();
        let version = calculated.version.fetch_add(0, Ordering::Acquire);
        let mut write = self.dependencies.write().unwrap();
        match write.entry(calculated.id) {
            Entry::Occupied(e) => {
                eprintln!("TODO - cyclic");
            }
            Entry::Vacant(e) => {
                e.insert((version, expr));
            }
        }
        calculated
    }

    /// NB: gradient calculation is outside of execution tracking
    fn grad(&self, calculated: Tensor<'env, F>, for_ultimate_input: Tensor<'env, F>) -> Option<Tensor<'env, F>> {
        if calculated.id == for_ultimate_input.id {
            //TODO optimize - move into the ops?


            // println!("TODO multi-dim");
            //TODO multi-dim? well-known ONE tensor?


            return Some(calculated.env.scalar(F::one()));
        }


        //TODO store for caching


        if let Some((version, expr)) = self.dependencies.read().unwrap().get(&calculated.id) {
            if *version != calculated.get_version() {
                // the calculated tensor was modified in place since the gradient was calculated
                todo!()
            }

            match expr {
                TrackerExpression::Single(t, op) => {
                    // 'single' means applying a function f to a (potentially calculated) inner
                    //   tensor g, with the constraint that f itself is independent of any tracked
                    //   variables. So the chain rule applies: ( f(g(x)) )' = f'(g(x) * g'(x)

                    // We start with g'(x) because it may turn out not to depend on the
                    //  variables we are interested in

                    //TODO avoid recursion
                    //TODO store / cache gradients?
                    let inner_grad = self.grad(t.r(), for_ultimate_input);

                    if let Some(inner_grad) = inner_grad {
                        //TODO assert that the inner gradient matches the number of ultimate input vars

                        return Some(op.grad(t.r(), inner_grad));
                        // let outer_grad = op.grad(t.r(), inner_grad);
                        // {
                        //     multiply outer and inner gradient per element, i.e. apply the
                        //      chain rule per element
                            // return Some(outer_grad._mult(inner_grad));
                        // }
                    }
                    else {
                        return None
                    }
                }
                TrackerExpression::Two(t1, t2, op) => {
                    let grad1 = self.grad(t1.r(), for_ultimate_input.r());
                    let grad2 = self.grad(t2.r(), for_ultimate_input);

                    return op.grad(t1.r(), grad1, t2.r(), grad2);
                }
            }
        }
        else {
            // the queried tensor was not calculated with this tracker -> end of traversal
            None
        }
    }
}

pub enum TrackerExpression<'env, F: Float> {
    Single(Tensor<'env, F>, Box<dyn SingleTensorOp<F>>),
    Two(Tensor<'env, F>, Tensor<'env, F>, Box<dyn TwoTensorOp<F>>),
}
impl <'env, F: Float> TrackerExpression<'env, F> {
    fn calc(&self) -> Tensor<'env, F> {
        match self {
            TrackerExpression::Single(t, op) => op.apply(t.r()),
            TrackerExpression::Two(t1, t2, op) => op.apply(t1.r(), t2.r())
        }
    }
}

pub trait SingleTensorOp<F: Float> {
    fn apply<'env>(&self, t: Tensor<'env, F>) -> Tensor<'env, F>;
    fn grad<'env>(&self, t: Tensor<'env, F>, t_grad: Tensor<'env, F>) -> Tensor<'env, F>;
}
pub trait TwoTensorOp<F: Float> {
    fn apply<'env>(&self, t1: Tensor<'env, F>, t2: Tensor<'env, F>) -> Tensor<'env, F>;
    fn grad<'env>(&self, t1: Tensor<'env, F>, grad1: Option<Tensor<'env, F>>, t2: Tensor<'env, F>, grad2: Option<Tensor<'env, F>>) -> Option<Tensor<'env, F>>;
}

pub struct PowInt {
    exp: u16
}
impl <F: Float> SingleTensorOp<F> for PowInt {
    //TODO limit to exp > 1 - create to other 'single' ops instead

    fn apply<'env>(&self, t: Tensor<'env, F>) -> Tensor<'env, F> {
        t._pow_int(self.exp)
    }
    fn grad<'env>(&self, t: Tensor<'env, F>, t_grad: Tensor<'env, F>) -> Tensor<'env, F> {
        let outer_grad = t._pow_int(self.exp - 1)._mult(t.env.scalar(self.exp.into()));
        outer_grad._mult(t_grad)
    }
}

pub struct MultOp{}
impl <F: Float> TwoTensorOp<F> for MultOp {
    fn apply<'env>(&self, t1: Tensor<'env, F>, t2: Tensor<'env, F>) -> Tensor<'env, F> {
        t1._mult(t2)
    }

    fn grad<'env>(&self, t1: Tensor<'env, F>, grad1: Option<Tensor<'env, F>>, t2: Tensor<'env, F>, grad2: Option<Tensor<'env, F>>) -> Option<Tensor<'env, F>> {
        let term1 = grad1.map(|grad1| grad1._mult(t2.r()));
        let term2 = grad2.map(|grad2| t1._mult(grad2));

        match (term1, term2) {
            (None, None) => None,
            (Some(t1), None) => Some(t1),
            (None, Some(t2)) => Some(t2),
            (Some(t1), Some(t2)) => Some(t1._plus(t2))
        }
    }
}

pub struct PlusOp{}
impl <F: Float> TwoTensorOp<F> for PlusOp {
    fn apply<'env>(&self, t1: Tensor<'env, F>, t2: Tensor<'env, F>) -> Tensor<'env, F> {
        t1._plus(t2)
    }

    fn grad<'env>(&self, t1: Tensor<'env, F>, grad1: Option<Tensor<'env, F>>, t2: Tensor<'env, F>, grad2: Option<Tensor<'env, F>>) -> Option<Tensor<'env, F>> {
        match (grad1, grad2) {
            (None, None) => None,
            (Some(t1), None) => Some(t1),
            (None, Some(t2)) => Some(t2),
            (Some(t1), Some(t2)) => Some(t1._plus(t2))
        }
    }
}

pub struct MinusOp{}
impl <F: Float> TwoTensorOp<F> for MinusOp {
    fn apply<'env>(&self, t1: Tensor<'env, F>, t2: Tensor<'env, F>) -> Tensor<'env, F> {
        t1._minus(t2)
    }

    fn grad<'env>(&self, t1: Tensor<'env, F>, grad1: Option<Tensor<'env, F>>, t2: Tensor<'env, F>, grad2: Option<Tensor<'env, F>>) -> Option<Tensor<'env, F>> {
        match (grad1, grad2) {
            (None, None) => None,
            (Some(t1), None) => Some(t1),
            (None, Some(t2)) => Some(t2),
            (Some(t1), Some(t2)) => Some(t1._minus(t2))
        }
    }
}

pub struct SumOp{}
impl <F: Float> SingleTensorOp<F> for SumOp {

    fn apply<'env>(&self, t: Tensor<'env, F>) -> Tensor<'env, F> {
        //TODO generalize to n-dim -> (n-1)-dim?
        t._sum()
    }

    fn grad<'env>(&self, _t: Tensor<'env, F>, t_grad: Tensor<'env, F>) -> Tensor<'env, F> {
        t_grad._sum()
    }
}


#[cfg(test)]
mod test {
    use std::f64::consts::PI;

    use rstest::*;

    use super::*;

    #[rstest]
    #[case(vec![0.0], 2, vec![0.0], vec![0.0])]
    #[case(vec![1.0], 2, vec![1.0], vec![2.0])]
    #[case(vec![2.0], 2, vec![4.0], vec![4.0])]
    #[case(vec![3.0], 2, vec![9.0], vec![6.0])]

    #[case(vec![0.0], 3, vec![0.0], vec![0.0])]
    #[case(vec![1.0], 3, vec![1.0], vec![3.0])]
    #[case(vec![2.0], 3, vec![8.0], vec![12.0])]
    #[case(vec![3.0], 3, vec![27.0], vec![27.0])]

    #[case(vec![0.0, 1.0, 2.0, 3.0], 2, vec![0.0, 1.0, 4.0,  9.0], vec![0.0, 2.0,  4.0,  6.0])]
    #[case(vec![0.0, 1.0, 2.0, 3.0], 3, vec![0.0, 1.0, 8.0, 27.0], vec![0.0, 3.0, 12.0, 27.0])]
    fn test_pow_int(#[case] x: Vec<f64>, #[case] pow: u16, #[case] y_expected: Vec<f64>, #[case] grad_expected: Vec<f64>) {
        assert_eq!(x.len(), y_expected.len());
        assert_eq!(x.len(), grad_expected.len());

        let env = TensorEnv::new();

        if x.len() == 1 {
            let tracker = RegularExecutionTracker::new();

            let x = env.scalar(x[0]);
            let y = x.pow_int(pow, &tracker);

            y.assert_is_pretty_much_equal_to(&env.scalar(y_expected[0]));
            tracker.grad(y.r(), x.r()).unwrap().assert_is_pretty_much_equal_to(&env.scalar(grad_expected[0]));
        }

        let tracker = RegularExecutionTracker::new();

        let x = env.vector(x);
        let y = x.pow_int(pow, &tracker);

        y.assert_is_pretty_much_equal_to(&env.vector(y_expected));
        tracker.grad(y.r(), x.r()).unwrap().assert_is_pretty_much_equal_to(&env.vector(grad_expected));
    }

    #[rstest]
    #[case(vec![1.0], 2.0, vec![1.0])]
    #[case(vec![1.0, 2.0, 3.0], 12.0, vec![6.0])]
    fn test_sum(#[case] x: Vec<f64>, #[case] y_expected: f64, #[case] grad_expected: Vec<f64>) {
        let env = TensorEnv::new();

        let q = env.scalar(2.0);

        if x.len() == 1 {
            let tracker = RegularExecutionTracker::new();

            let x = env.scalar(x[0]).mult(q.r(), &tracker);
            let y = x.sum(&tracker);
            y.assert_is_pretty_much_equal_to(&env.scalar(y_expected));
            tracker.grad(y, q.r()).unwrap().assert_is_pretty_much_equal_to(&env.scalar(grad_expected[0]));
        }

        let tracker = RegularExecutionTracker::new();

        let x = env.vector(x).mult(q.r(), &tracker);
        let y = x.sum(&tracker);
        y.assert_is_pretty_much_equal_to(&env.scalar(y_expected));
        tracker.grad(y, q).unwrap().assert_is_pretty_much_equal_to(&env.vector(grad_expected));
    }

    // #[test]
    // fn test_play_around() {
    //     let env = TensorEnv::new();
    //
    //     let a = env.scalar(2.0);
    //     let b = env.scalar(3.0);
    //     let c = env.scalar(5.0);
    //
    //     let f = a._mult_with_scalar(b);
    //     let g = f._plus(c);
    //
    //     println!("{:?}", g);
    // }
    //
    // #[test]
    // fn test_gradient() {
    //     let env = TensorEnv::new();
    //
    //     let a = env.scalar(1.0);
    //     let b = env.scalar(3.0);
    //     let c = env.scalar(5.0);
    //
    //     let x = env.scalar(2.0);
    //
    //     let tracker = RegularExecutionTracker::new();
    //     // let y = a.plus(x.r(), &tracker);
    //     let y = a.plus(
    //         b.mult_element_wise(x.r(), &tracker).plus(
    //             c.mult_element_wise(
    //                 x.r().pow_int(2, &tracker),
    //                 &tracker,
    //             ),
    //             &tracker
    //         ),
    //         &tracker
    //     );
    //     // a + b*x + c*x^2   @ a=1, b=3, c=5, x=2
    //     println!("{:?}", y);
    //     println!("dy/da: {:?}", tracker.grad(y.r(), a.r()));
    //     println!("dy/db: {:?}", tracker.grad(y.r(), b.r()));
    //     println!("dy/dc: {:?}", tracker.grad(y.r(), c.r()));
    // }
    //
    #[test]
    fn test_gradient_opt() {
        let env = TensorEnv::new();

        let a = env.scalar(0.1);
        let b = env.scalar(0.9);
        let c = env.scalar(0.1);
        let d = env.scalar(-0.1);

        let x = env.random_lin(-PI, PI, 5_000);
        let y = x._sin();

        let learning_rate = env.scalar(1e-6);

        for t in 0..2_000 {
            let tracker = RegularExecutionTracker::new();

            let y3 = x.r()
                .pow_int(3, &tracker)
                .mult(d.r(), &tracker);
            let y2 = x.r()
                .pow_int(2, &tracker)
                .mult(c.r(), &tracker);
            let y1 = x.mult(b.r(), &tracker);
            let y_pred = y3
                .plus(y2.r(), &tracker)
                .plus(y1.r(), &tracker)
                .plus(a.r(), &tracker);

            let loss = y_pred.minus(y.r(), &tracker).pow_int(2, &tracker).sum(&tracker);

            let grad_a = tracker.grad(loss.r(), a.r()).unwrap();
            let grad_b = tracker.grad(loss.r(), b.r()).unwrap();
            let grad_c = tracker.grad(loss.r(), c.r()).unwrap();
            let grad_d = tracker.grad(loss.r(), d.r()).unwrap();

            a._sub_in_place(grad_a.r()._mult(learning_rate.r()));
            b._sub_in_place(grad_b.r()._mult(learning_rate.r()));
            c._sub_in_place(grad_c.r()._mult(learning_rate.r()));
            d._sub_in_place(grad_d.r()._mult(learning_rate.r()));

            if t%100 == 0 {
                // println!("loss {:?}: {:?}, {:?}  --  {:?}, {:?}", loss, a.r(), b.r(), grad_a, grad_b);
                println!("loss {:?}: {:?}, {:?}, {:?}, {:?}  --  {:?}, {:?}, {:?}, {:?}", loss, a.r(), b.r(), c.r(), d.r(), grad_a, grad_b, grad_c, grad_d);
            }
        }
    }
}





