use std::cmp::max;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::fmt::{Debug, Display, Formatter};
use std::ops::{Add, Div, Mul, Neg, Sub, SubAssign};
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::atomic::Ordering::Acquire;

use lazy_static::lazy_static;
use rand::random;
use rustc_hash::FxHashMap;
use triomphe::Arc;

//TODO feature flag
macro_rules! trace {
    ($fmt:literal $(, $e:expr)*) => {
        // log::trace!($fmt $(, $e)*)
    }
}


pub trait Float: 'static +
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

#[derive(Clone)]
pub struct TensorEnv<F: Float> {
    //TODO why doesn't this work with triomphe?
    implicit_tracker: Arc<RwLock<std::sync::Arc<dyn ExecutionTracker<F>>>>,
}
impl <F: Float> TensorEnv<F> {
    pub fn new() -> TensorEnv<F> {
        TensorEnv {
            implicit_tracker: Arc::new(RwLock::new(std::sync::Arc::new(NoTracker{}))),
        }
    }

    fn assemble(&self, geometry: Geometry, data: Vec<F>) -> Tensor<F> {
        let id = TENSOR_ID_COUNTER.fetch_add(1, Ordering::SeqCst);
        Tensor {
            env: self.clone(),
            id,
            version: Default::default(),
            geometry,
            data: Arc::new(RwLock::new(data)),
        }
    }

    pub fn scalar(&self, value: F) -> Tensor<F> {
        self.assemble(Geometry::scalar(),vec![value])
    }

    pub fn zero(&self) -> Tensor<F> {
        self.scalar(F::zero())
    }
    pub fn one(&self) -> Tensor<F> {
        self.scalar(F::one())
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

    pub fn with_tracker(&self, code: impl FnOnce(&dyn ExecutionTracker<F>)) {
        let prev = self.implicit_tracker.read().unwrap().clone();

        let tracker = std::sync::Arc::new(RegularExecutionTracker::new());
        *self.implicit_tracker.write().unwrap() = tracker.clone();

        code(tracker.as_ref());

        *self.implicit_tracker.write().unwrap() = prev;
    }

    pub fn untracked(&self, code: impl FnOnce()) {
        let prev = self.implicit_tracker.read().unwrap().clone();
        *self.implicit_tracker.write().unwrap() = std::sync::Arc::new(NoTracker{});

        code();

        *self.implicit_tracker.write().unwrap() = prev;
    }

    pub fn current_tracker(&self) -> std::sync::Arc<dyn ExecutionTracker<F>> {
        self.implicit_tracker.read().unwrap().clone()
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
pub struct Tensor<F: Float> {
    //TODO optional name (for debugging)
    env: TensorEnv<F>,
    id: u32,
    /// incremented on each change
    version: Arc<AtomicU32>,
    geometry: Geometry,
    data: Arc<RwLock<Vec<F>>>,
}
impl<F: Float> Debug for Tensor<F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}: {:?}", self.geometry, self.read())
    }
}

impl <F: Float> Add for Tensor<F> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self.plus_full(rhs, self.env.current_tracker().as_ref())
    }
}
impl <F: Float> Sub for Tensor<F> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self.minus_full(rhs, self.env.current_tracker().as_ref())
    }
}
impl <F: Float> SubAssign for Tensor<F> {
    fn sub_assign(&mut self, rhs: Self) {
        //TODO tracking etc.
        self._sub_in_place(rhs);
    }
}

impl <F: Float> Mul for Tensor<F> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self.mult_full(rhs, self.env.current_tracker().as_ref())
    }
}
impl <F: Float> Mul<F> for Tensor<F> { //TODO F * T<F>
    type Output = Self;

    fn mul(self, rhs: F) -> Self::Output {
        self.mult_full(self.env.scalar(rhs), self.env.current_tracker().as_ref())
    }
}


impl<F: Float> Tensor<F> {
    pub fn r(&self) -> Tensor<F> {
        self.clone()
    }

    #[inline]
    pub fn is_zero(&self) -> bool {
        self.geometry == Geometry::scalar() && self.read()[0] == F::zero()
    }

    #[inline]
    pub fn is_one(&self) -> bool {
        self.geometry == Geometry::scalar() && self.read()[0] == F::one()
    }

    /// Comparing floating point numbers for exact equality does not work much of the time due to
    ///  rounding errors. This is a convenience function that compares two tensors element by
    ///  element, returning true if each of the elements are equal within a heuristically chosen
    ///  EPSILON.
    ///
    /// This method is intended as convenience functionality for unit tests.
    pub fn is_pretty_much_equal_to(&self, other: &Tensor<F>) -> bool {
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

    pub fn assert_is_pretty_much_equal_to(&self, other: &Tensor<F>) {
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
    fn binary_operation(&self, other: &Tensor<F>, f: impl Fn(&[F], &[F], &mut Vec<F>)) -> Tensor<F> {
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


    pub fn plus_full(&self, other: Tensor<F>, tracker: &dyn ExecutionTracker<F>) -> Tensor<F> {
        tracker.calc(TrackerExpression::Two(self.r(), other.r(), Box::new(PlusOp{})))
    }
    fn _plus(&self, rhs: Tensor<F>) -> Tensor<F> {
        trace!("_plus ({:?}+{:?})", self.geometry, rhs.geometry);

        if self.is_zero() {
            return rhs; //TODO unit test
        }
        if rhs.is_zero() {
            return self.r();
        }

        self.binary_operation(&rhs, |a, b, result| {
            for i in 0..a.len() {
                result.push(a[i] + b[i]);
            }
        })
    }

    pub fn minus_full(&self, other: Tensor<F>, tracker: &dyn ExecutionTracker<F>) -> Tensor<F> {
        tracker.calc(TrackerExpression::Two(self.r(), other, Box::new(MinusOp{})))
    }
    fn _minus(&self, rhs: Tensor<F>) -> Tensor<F> {
        trace!("_minus ({:?}-{:?})", self.geometry, rhs.geometry);

        if rhs.is_zero() {
            return self.r();
        }

        self.binary_operation(&rhs, |a, b, result| {
            for i in 0..a.len() {
                result.push(a[i] - b[i]);
            }
        })
    }

    fn sum(&self) -> Tensor<F> {
        self.sum_full(self.env.current_tracker().as_ref())
    }
    fn sum_full(&self, tracker: &dyn ExecutionTracker<F>) -> Tensor<F> {
        tracker.calc(TrackerExpression::Single(self.r(), Box::new(SumOp{}))) //TODO convenience API
    }
    fn _sum(&self) -> Tensor<F> {
        trace!("_sum ({:?})", self.geometry);
        let mut result = F::zero();

        for &x in self.read().iter() {
            result = result + x;
        }
        self.env.scalar(result)
    }

    pub fn mult_full(&self, rhs: Tensor<F>, tracker: &dyn ExecutionTracker<F>) -> Tensor<F> {
        tracker.calc(TrackerExpression::Two(self.r(), rhs, Box::new(MultOp{})))
    }
    fn _mult(&self, rhs: Tensor<F>) -> Tensor<F> {
        trace!("_mult ({:?}*{:?})", self.geometry, rhs.geometry);

        if self.is_one() {
            return rhs;
        }
        if rhs.is_one() {
            return self.r();
        }
        if self.is_zero() || rhs.is_zero() {
            return self.env.scalar(F::zero());
        }

        self.binary_operation(&rhs, |a, b, result| {
            for i in 0..a.len() {
                result.push(a[i] * b[i]);
            }
        })
    }

    pub fn pow(&self, exp: u16) -> Tensor<F> {
        self.pow_int_full(exp, self.env.current_tracker().as_ref())
    }
    pub fn pow_int_full(&self, exp: u16, tracker: &dyn ExecutionTracker<F>) -> Tensor<F> {
        tracker.calc(TrackerExpression::Single(self.r(), Box::new(PowInt { exp })))
    }
    pub fn _pow_int(&self, exp: u16) -> Tensor<F> {
        trace!("_pow_int ({:?}^{})", self.geometry, exp);

        if exp == 0 {
            return self.env.scalar(F::one());
        }
        if exp == 1 {
            return self.r();
        }

        let v1 = self.read();
        let mut data = Vec::with_capacity(v1.len());
        for &x in v1.iter() {
            match exp {
                0 => panic!("should have been handled by short-circuit logic"),
                1 => panic!("should have been handled by short-circuit logic"),
                2 => data.push(x*x),
                3 => data.push(x*x*x),
                4 => data.push(x*x*x*x), //TODO unit test
                _ => data.push(x.pow_i(exp)), //TODO up to what exponent is multiplying more efficient?
            }
        }
        self.env.assemble(self.geometry.clone(), data)
    }

    //TODO sin()
    pub fn _sin(&self) -> Tensor<F> {
        let v1 = self.read();
        let mut data = Vec::with_capacity(v1.len());
        for &x in v1.iter() {
            data.push(x.sin());
        }
        self.env.assemble(self.geometry.clone(), data)
    }

    pub fn _sub_in_place(&self, other: Tensor<F>) {
        //TODO assert / lift geometry
        let mut data = self.write();
        let other = other.read();

        for i in 0..data.len() {
            data[i] = data[i] - other[i];
        }
    }
}

pub trait ExecutionTracker<F: Float> {
    fn calc(&self, expr: TrackerExpression<F>) -> Tensor<F>;

    fn grad(&self, calculated: &Tensor<F>, for_ultimate_input: &Tensor<F>) -> Tensor<F>;
}

pub struct NoTracker {}
impl <F: Float> ExecutionTracker<F> for NoTracker {
    fn calc(&self, expr: TrackerExpression<F>) -> Tensor<F> {
        expr.calc()
    }

    fn grad(&self, _calculated: &Tensor<F>, _for_ultimate_input: &Tensor<F>) -> Tensor<F> {
        _calculated.env.zero()
    }
}

pub struct RegularExecutionTracker<F: Float> {
    dependencies: RwLock<HashMap<u32, (u32, TrackerExpression<F>)>>,
}
impl <F: Float> RegularExecutionTracker<F> {
    pub fn new() -> RegularExecutionTracker<F> {
        RegularExecutionTracker {
            dependencies: Default::default()
        }
    }
}
impl <F: Float> ExecutionTracker<F> for RegularExecutionTracker<F> {
    fn calc(&self, expr: TrackerExpression<F>) -> Tensor<F> {
        let calculated = expr.calc();
        let version = calculated.version.fetch_add(0, Ordering::Acquire);
        let mut write = self.dependencies.write().unwrap();
        match write.entry(calculated.id) {
            Entry::Occupied(_) => {
                eprintln!("TODO - cyclic");
            }
            Entry::Vacant(e) => {
                e.insert((version, expr));
            }
        }
        calculated
    }

    /// NB: gradient calculation is outside of execution tracking
    fn grad(&self, calculated: &Tensor<F>, for_ultimate_input: &Tensor<F>) -> Tensor<F> {
        trace!("{}grad [{:?}]", indent, calculated.id);

        let mut grad_cache: FxHashMap<u32, Tensor<F>> = Default::default();
        grad_cache.insert(for_ultimate_input.id, for_ultimate_input.env.one()); // pre-filling for termination

        let mut worklist = vec![calculated.r()];

        while let Some(cur) = worklist.pop() {
            let cur_grad: Tensor<F> = if let Some((version, expr)) = self.dependencies.read().unwrap().get(&cur.id) {
                if *version != cur.get_version() {
                    // the calculated tensor was modified in place since the gradient was calculated
                    todo!()
                }
                match expr {
                    TrackerExpression::Single(t, op) => {
                        trace!("{}single: {:?}", worklist.len(), op);

                        // 'single' means applying a function f to a (potentially calculated) inner
                        //   tensor g, with the constraint that f itself is independent of any tracked
                        //   variables. So the chain rule applies: ( f(g(x)) )' = f'(g(x) * g'(x)

                        // We start with g'(x) because it may turn out not to depend on the
                        //  variables we are interested in

                        if let Some(inner_grad) = grad_cache.get(&t.id) {
                            trace!("after inner_grad [{:?}]", calculated.id);
                            if inner_grad.is_zero() {
                                inner_grad.r()
                            } else {
                                op.grad(t.r(), inner_grad.r())
                            }
                        }
                        else {
                            worklist.push(cur);
                            worklist.push(t.r());
                            continue;
                        }
                    }
                    TrackerExpression::Two(t1, t2, op) => {
                        trace!("{}two: {:?}", worklist.len(), op);
                        match (grad_cache.get(&t1.id), grad_cache.get(&t2.id)) {
                            (Some(grad1), Some(grad2)) => {
                                op.grad(t1.r(), grad1.r(), t2.r(), grad2.r())
                            }
                            (grad1, grad2) => {
                                worklist.push(cur);
                                if grad1.is_none() {
                                    worklist.push(t1.r());
                                }
                                if grad2.is_none() {
                                    worklist.push(t2.r());
                                }
                                continue;
                            }
                        }
                    }
                }
            }
            else {
                // some part of the calculation that was not tracked by this tracker
                //TODO logging
                cur.env.zero()
            };

            if cur.id == calculated.id {
                return cur_grad.r();
            }
            grad_cache.insert(cur.id, cur_grad);
        }

        panic!("internal error in gradient calculation")
    }
}

pub enum TrackerExpression<F: Float> {
    Single(Tensor<F>, Box<dyn SingleTensorOp<F>>),
    Two(Tensor<F>, Tensor<F>, Box<dyn TwoTensorOp<F>>),
}
impl <F: Float> TrackerExpression<F> {
    fn calc(&self) -> Tensor<F> {
        match self {
            TrackerExpression::Single(t, op) => op.apply(t.r()),
            TrackerExpression::Two(t1, t2, op) => op.apply(t1.r(), t2.r())
        }
    }
}

pub trait SingleTensorOp<F: Float>: Debug {
    fn apply<'env>(&self, t: Tensor<F>) -> Tensor<F>;
    fn grad<'env>(&self, t: Tensor<F>, t_grad: Tensor<F>) -> Tensor<F>;
}
pub trait TwoTensorOp<F: Float>: Debug {
    fn apply<'env>(&self, t1: Tensor<F>, t2: Tensor<F>) -> Tensor<F>;
    fn grad<'env>(&self, t1: Tensor<F>, grad1: Tensor<F>, t2: Tensor<F>, grad2: Tensor<F>) -> Tensor<F>;
}

#[derive(Debug)]
pub struct PowInt {
    exp: u16
}
impl <F: Float> SingleTensorOp<F> for PowInt {
    //TODO limit to exp > 1 - create to other 'single' ops instead

    fn apply<'env>(&self, t: Tensor<F>) -> Tensor<F> {
        t._pow_int(self.exp)
    }
    fn grad<'env>(&self, t: Tensor<F>, t_grad: Tensor<F>) -> Tensor<F> {
        let pow_grad = t._pow_int(self.exp - 1)._mult(t.env.scalar(self.exp.into()));
        pow_grad._mult(t_grad)
    }
}

#[derive(Debug)]
pub struct MultOp{}
impl <F: Float> TwoTensorOp<F> for MultOp {
    fn apply<'env>(&self, t1: Tensor<F>, t2: Tensor<F>) -> Tensor<F> {
        t1._mult(t2)
    }

    fn grad<'env>(&self, t1: Tensor<F>, grad1: Tensor<F>, t2: Tensor<F>, grad2: Tensor<F>) -> Tensor<F> {
        let term1 = grad1._mult(t2);
        let term2 = t1._mult(grad2);

        term1._plus(term2)
    }
}

#[derive(Debug)]
pub struct PlusOp{}
impl <F: Float> TwoTensorOp<F> for PlusOp {
    fn apply<'env>(&self, t1: Tensor<F>, t2: Tensor<F>) -> Tensor<F> {
        t1._plus(t2)
    }

    fn grad<'env>(&self, _t1: Tensor<F>, grad1: Tensor<F>, _t2: Tensor<F>, grad2: Tensor<F>) -> Tensor<F> {
        grad1._plus(grad2)
    }
}

#[derive(Debug)]
pub struct MinusOp{}
impl <F: Float> TwoTensorOp<F> for MinusOp {
    fn apply<'env>(&self, t1: Tensor<F>, t2: Tensor<F>) -> Tensor<F> {
        t1._minus(t2)
    }

    fn grad<'env>(&self, _t1: Tensor<F>, grad1: Tensor<F>, _t2: Tensor<F>, grad2: Tensor<F>) -> Tensor<F> {
        grad1._minus(grad2)
    }
}

#[derive(Debug)]
pub struct SumOp{}
impl <F: Float> SingleTensorOp<F> for SumOp {

    fn apply<'env>(&self, t: Tensor<F>) -> Tensor<F> {
        //TODO generalize to n-dim -> (n-1)-dim?
        t._sum()
    }

    fn grad<'env>(&self, _t: Tensor<F>, t_grad: Tensor<F>) -> Tensor<F> {
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
            let y = x.pow_int_full(pow, &tracker);

            y.assert_is_pretty_much_equal_to(&env.scalar(y_expected[0]));
            tracker.grad(&y, &x).assert_is_pretty_much_equal_to(&env.scalar(grad_expected[0]));
        }

        let tracker = RegularExecutionTracker::new();

        let x = env.vector(x);
        let y = x.pow_int_full(pow, &tracker);

        y.assert_is_pretty_much_equal_to(&env.vector(y_expected));
        tracker.grad(&y, &x).assert_is_pretty_much_equal_to(&env.vector(grad_expected));
    }

    #[rstest]
    #[case(vec![1.0], 2.0, vec![1.0])]
    #[case(vec![1.0, 2.0, 3.0], 12.0, vec![6.0])]
    fn test_sum(#[case] x: Vec<f64>, #[case] y_expected: f64, #[case] grad_expected: Vec<f64>) {
        let env = TensorEnv::new();
        let q = env.scalar(2.0);

        if x.len() == 1 {
            let tracker = RegularExecutionTracker::new();

            let x = env.scalar(x[0]).mult_full(q.r(), &tracker);
            let y = x.sum_full(&tracker);
            y.assert_is_pretty_much_equal_to(&env.scalar(y_expected));
            tracker.grad(&y, &q).assert_is_pretty_much_equal_to(&env.scalar(grad_expected[0]));
        }

        let tracker = RegularExecutionTracker::new();

        let x = env.vector(x).mult_full(q.r(), &tracker);
        let y = x.sum_full(&tracker);
        y.assert_is_pretty_much_equal_to(&env.scalar(y_expected));
        tracker.grad(&y, &q).assert_is_pretty_much_equal_to(&env.vector(grad_expected));
    }

    fn vec_to_tensor(env: &TensorEnv<f64>, x: Vec<f64>) -> Tensor<f64> {
        if x.len() == 1 {
            env.scalar(x[0])
        }
        else {
            env.vector(x)
        }
    }

    #[rstest]
    #[case(vec![1.0], vec![2.0], vec![8.0], vec![2.0], vec![3.0])]
    #[case(vec![1.0, 1.0, 4.0], vec![2.0, 3.0, 7.0], vec![8.0, 11.0, 29.0], vec![2.0], vec![3.0])]
    #[case(vec![1.0], vec![2.0, 3.0, 7.0], vec![8.0, 11.0, 23.0], vec![2.0], vec![3.0])]
    #[case(vec![1.0, 1.0, 4.0], vec![2.0], vec![8.0, 8.0, 14.0], vec![2.0], vec![3.0])]
    fn test_plus(#[case] a: Vec<f64>, #[case] b: Vec<f64>, #[case] sum: Vec<f64>, #[case] grad_a: Vec<f64>, #[case] grad_b: Vec<f64>) {
        let env = TensorEnv::new();
        let q1 = env.scalar(2.0);
        let q2 = env.scalar(3.0);

        let sum = env.vector(sum);
        let grad_a = env.vector(grad_a);
        let grad_b = env.vector(grad_b);

        let tracker = RegularExecutionTracker::new();

        let a = vec_to_tensor(&env, a);
        let b = vec_to_tensor(&env, b);

        let s1 = a.r().mult_full(q1.r(), &tracker);
        let s2 = b.r().mult_full(q2.r(), &tracker);
        let y = s1.plus_full(s2, &tracker);
        y.assert_is_pretty_much_equal_to(&sum);
        tracker.grad(&y, &a).assert_is_pretty_much_equal_to(&grad_a);
        tracker.grad(&y, &b).assert_is_pretty_much_equal_to(&grad_b);
    }

    #[rstest]
    #[case(vec![1.0], vec![2.0], vec![-4.0], vec![2.0], vec![-3.0])]
    #[case(vec![1.0, 1.0, 4.0], vec![2.0, 3.0, 7.0], vec![-4.0, -7.0, -13.0], vec![2.0], vec![-3.0])]
    #[case(vec![1.0], vec![2.0, 3.0, 7.0], vec![-4.0, -7.0, -19.0], vec![2.0], vec![-3.0])]
    #[case(vec![1.0, 1.0, 4.0], vec![2.0], vec![-4.0, -4.0, 2.0], vec![2.0], vec![-3.0])]
    fn test_minus(#[case] a: Vec<f64>, #[case] b: Vec<f64>, #[case] sum: Vec<f64>, #[case] grad_a: Vec<f64>, #[case] grad_b: Vec<f64>) {
        let env = TensorEnv::new();
        let q1 = env.scalar(2.0);
        let q2 = env.scalar(3.0);

        let sum = env.vector(sum);
        let grad_a = env.vector(grad_a);
        let grad_b = env.vector(grad_b);

        let tracker = RegularExecutionTracker::new();

        let a = vec_to_tensor(&env, a);
        let b = vec_to_tensor(&env, b);

        let s1 = a.r().mult_full(q1.r(), &tracker);
        let s2 = b.r().mult_full(q2.r(), &tracker);
        let y = s1.minus_full(s2, &tracker);
        y.assert_is_pretty_much_equal_to(&sum);
        tracker.grad(&y, &a).assert_is_pretty_much_equal_to(&grad_a);
        tracker.grad(&y, &b).assert_is_pretty_much_equal_to(&grad_b);
    }

    #[rstest]
    #[case(vec![1.0], vec![2.0], vec![12.0], vec![12.0], vec![6.0])]
    #[case(vec![1.0, 1.0, 4.0], vec![2.0, 3.0, 7.0], vec![12.0, 18.0, 168.0], vec![12.0, 18.0, 42.0], vec![6.0, 6.0, 24.0])]
    #[case(vec![1.0], vec![2.0, 3.0, 7.0], vec![12.0, 18.0, 42.0], vec![12.0, 18.0, 42.0], vec![6.0])]
    #[case(vec![1.0, 1.0, 4.0], vec![2.0], vec![12.0, 12.0, 48.0], vec![12.0], vec![6.0, 6.0, 24.0])]
    fn test_mult(#[case] a: Vec<f64>, #[case] b: Vec<f64>, #[case] prod: Vec<f64>, #[case] grad_a: Vec<f64>, #[case] grad_b: Vec<f64>) {
        // q1(x) = 2*x
        // q2(x) = 3*x
        // r(x1,x2) = x1*x2
        //
        // f(a, b) = f(q1(a), q2(b))
        //         = 2*a * 3*b
        // df/da   = q1'*q2 + q1*q2'
        //         = 2 * 3*b + 2*0
        //         = 2 * 3*b

        let env = TensorEnv::new();
        let q1 = env.scalar(2.0);
        let q2 = env.scalar(3.0);

        let product = env.vector(prod);
        let grad_a = env.vector(grad_a);
        let grad_b = env.vector(grad_b);

        let tracker = RegularExecutionTracker::new();

        let a = vec_to_tensor(&env, a);
        let b = vec_to_tensor(&env, b);

        let s1 = a.r().mult_full(q1.r(), &tracker);
        let s2 = b.r().mult_full(q2.r(), &tracker);
        let y = s1.mult_full(s2, &tracker);
        y.assert_is_pretty_much_equal_to(&product);
        tracker.grad(&y, &a).assert_is_pretty_much_equal_to(&grad_a);
        tracker.grad(&y, &b).assert_is_pretty_much_equal_to(&grad_b);
    }

    #[test]
    fn test_gradient_explicit() {
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
                .pow_int_full(3, &tracker)
                .mult_full(d.r(), &tracker);
            let y2 = x.r()
                .pow_int_full(2, &tracker)
                .mult_full(c.r(), &tracker);
            let y1 = x.mult_full(b.r(), &tracker);
            let y_pred = y3
                .plus_full(y2, &tracker)
                .plus_full(y1, &tracker)
                .plus_full(a.r(), &tracker);

            let loss = y_pred.minus_full(y.r(), &tracker).pow_int_full(2, &tracker).sum_full(&tracker);

            trace!("-------- before");
            let grad_a = tracker.grad(&loss, &a);
            let grad_b = tracker.grad(&loss, &b);
            let grad_c = tracker.grad(&loss, &c);
            let grad_d = tracker.grad(&loss, &d);

            a._sub_in_place(grad_a.r()._mult(learning_rate.r()));
            b._sub_in_place(grad_b.r()._mult(learning_rate.r()));
            c._sub_in_place(grad_c.r()._mult(learning_rate.r()));
            d._sub_in_place(grad_d.r()._mult(learning_rate.r()));
            trace!("-------- after");

            if t%100 == 0 {
                println!("loss {:?}: {:?}, {:?}, {:?}, {:?}  --  {:?}, {:?}, {:?}, {:?}", loss, a.r(), b.r(), c.r(), d.r(), grad_a, grad_b, grad_c, grad_d);
            }
        }
    }

    #[test]
    fn test_gradient_implicit_tracker() {
        let env = TensorEnv::new();

        let mut a = env.scalar(0.1);
        let mut b = env.scalar(0.9);
        let mut c = env.scalar(0.1);
        let mut d = env.scalar(-0.1);

        let x = env.random_lin(-PI, PI, 5_000);
        let y = x._sin();

        let learning_rate = 1e-6;

        for t in 0..2_000 {
            env.with_tracker(|tracker| {
                let y3 = d.r() * x.pow(3);
                let y2 = c.r() * x.pow(2);
                let y1 = b.r() * x.r();
                let y_pred = y3 + y2 + y1 + a.r();

                let loss = (y_pred - y.r()).pow(2).sum();

                let grad_a = tracker.grad(&loss, &a);
                let grad_b = tracker.grad(&loss, &b);
                let grad_c = tracker.grad(&loss, &c);
                let grad_d = tracker.grad(&loss, &d);

                env.untracked(|| {
                    a -= grad_a * learning_rate;
                    b -= grad_b * learning_rate;
                    c -= grad_c * learning_rate;
                    d -= grad_d * learning_rate;
                });

                if t%100 == 0 {
                    println!("loss {:?}: {:?}, {:?}, {:?}, {:?}", loss, a, b, c, d);
                }
            });
        }
    }
}

