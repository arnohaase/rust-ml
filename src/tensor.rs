use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::fmt::{Debug, Display, Formatter};
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Sub};
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::atomic::Ordering::Acquire;
use lazy_static::lazy_static;
use triomphe::Arc;

pub trait Float:
    Copy + Display + Debug +
    Add<Output=Self> + Sub<Output=Self> + Mul<Output=Self> + Div<Output=Self> +
    From<u16>
{
    fn one() -> Self;
}

impl Float for f64 {
    fn one() -> f64 {
        1.0
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

    pub fn create_scalar(&self, value: F) -> Tensor<F> {
        self.assemble(Geometry::Scalar,vec![value])
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Geometry { //TODO
    Scalar,
    Vector(usize),
    // Matrix,
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

    fn get_version(&self) -> u32 {
        self.version.fetch_add(0, Acquire)
    }

    fn read(&self) -> RwLockReadGuard<'_, Vec<F>> {
        self.data.read().unwrap()
    }
    fn write(&self) -> RwLockWriteGuard<'_, Vec<F>> {
        self.data.write().unwrap()
    }


    pub fn plus(&self, other: Tensor<'env, F>, tracker: &dyn ExecutionTracker<'env, F>) -> Tensor<'env, F> {
        tracker.calc(TrackerExpression::Two(self.r(), other, Box::new(PlusOp{})))
    }
    fn _plus(&self, other: Tensor<'_, F>) -> Tensor<'env, F> {
        assert_eq!(self.geometry, other.geometry);
        let v1 = self.read();
        let v2 = other.read();

        let mut data = Vec::with_capacity(v1.len());
        for i in 0..v1.len() {
            data.push(v1[i] + v2[i]);
        }
        self.env.assemble(self.geometry, data)
    }

    pub fn mult_with_scalar(&self, scalar: Tensor<'env, F>, tracker: &dyn ExecutionTracker<'env, F>) -> Tensor<'env, F> { //TODO generalize - this is just to get an initial example to work
        tracker.calc(TrackerExpression::Two(self.r(), scalar, Box::new(MultWithScalarOp{})))
    }
    fn _mult_with_scalar(&self, scalar: Tensor<'_, F>) -> Tensor<'env, F> { //TODO generalize - this is just to get an initial example to work
        assert_eq!(scalar.geometry, Geometry::Scalar);

        let scalar = scalar.read()[0];
        let v1 = self.read();

        let mut data = Vec::with_capacity(v1.len());
        for &x in v1.iter() {
            data.push(x*scalar);
        }
        self.env.assemble(self.geometry, data)
    }

    pub fn mult_element_wise(&self, rhs: Tensor<'env, F>, tracker: &dyn ExecutionTracker<'env, F>) -> Tensor<'env, F> { //TODO generalize - this is just to get an initial example to work
        tracker.calc(TrackerExpression::Two(self.r(), rhs, Box::new(MultElementWiseOp{})))
    }

    fn _mult_element_wise(&self, rhs: Tensor<'_, F>) -> Tensor<'env, F> { //TODO generalize - this is just to get an initial example to work
        assert_eq!(self.geometry, rhs.geometry);
        let v1 = self.read();
        let v2 = rhs.read();

        let mut data = Vec::with_capacity(v1.len());
        for i in 0..v1.len() {
            data.push(v1[i] * v2[i]);
        }

        self.env.assemble(self.geometry, data)
    }

    pub fn pow_int(&self, exp: u16, tracker: &dyn ExecutionTracker<'env, F>) -> Tensor<'env, F> {
        tracker.calc(TrackerExpression::Single(self.r(), Box::new(PowInt(exp))))
    }
    pub fn _pow_int(&self, exp: u16) -> Tensor<'env, F> {
        let v1 = self.read();
        let mut data = Vec::with_capacity(v1.len());
        for &x in v1.iter() {
            let mut value = x;
            for _ in 1..exp { //TODO this is hacky and inefficient, to get an example running early on
                value = value * x;
            }
            data.push(value);
        }
        self.env.assemble(self.geometry, data)
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


            return Some(calculated.env.create_scalar(F::one()));
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

                        let outer_grad = op.grad(t.r());
                        assert_eq!(inner_grad.geometry, outer_grad.geometry, "outer and inner gradients have different geometries, this is a bug in rust-ml");

                        // multiply outer and inner gradient per element, i.e. apply the
                        //  chain rule per element
                        return Some(outer_grad._mult_element_wise(inner_grad));
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

enum TrackerExpression<'env, F: Float> {
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
    fn grad<'env>(&self, t: Tensor<'env, F>) -> Tensor<'env, F>;
}
pub trait TwoTensorOp<F: Float> {
    fn apply<'env>(&self, t1: Tensor<'env, F>, t2: Tensor<'env, F>) -> Tensor<'env, F>;
    fn grad<'env>(&self, t1: Tensor<'env, F>, grad1: Option<Tensor<'env, F>>, t2: Tensor<'env, F>, grad2: Option<Tensor<'env, F>>) -> Option<Tensor<'env, F>>;
}

pub struct PowInt(u16);
impl <F: Float> SingleTensorOp<F> for PowInt {
    //TODO limit to exp > 1 - create to other 'single' ops instead

    fn apply<'env>(&self, t: Tensor<'env, F>) -> Tensor<'env, F> {
        t._pow_int(self.0)
    }
    fn grad<'env>(&self, t: Tensor<'env, F>) -> Tensor<'env, F> {
        t._pow_int(self.0 - 1)._mult_with_scalar(t.env.create_scalar(self.0.into()))
    }
}

pub struct MultElementWiseOp{}
impl <F: Float> TwoTensorOp<F> for MultElementWiseOp {
    fn apply<'env>(&self, t1: Tensor<'env, F>, t2: Tensor<'env, F>) -> Tensor<'env, F> {
        t1._mult_element_wise(t2)
    }

    fn grad<'env>(&self, t1: Tensor<'env, F>, grad1: Option<Tensor<'env, F>>, t2: Tensor<'env, F>, grad2: Option<Tensor<'env, F>>) -> Option<Tensor<'env, F>> {
        let term1 = grad1.map(|grad1| grad1._mult_element_wise(t2.r()));
        let term2 = grad2.map(|grad2| t1._mult_element_wise(grad2));

        match (term1, term2) {
            (None, None) => None,
            (Some(t1), None) => Some(t1),
            (None, Some(t2)) => Some(t2),
            (Some(t1), Some(t2)) => Some(t1._plus(t2))
        }
    }
}

pub struct MultWithScalarOp{}
impl <F: Float> TwoTensorOp<F> for MultWithScalarOp {
    fn apply<'env>(&self, t1: Tensor<'env, F>, t2: Tensor<'env, F>) -> Tensor<'env, F> {
        t1._mult_with_scalar(t2)
    }

    fn grad<'env>(&self, t1: Tensor<'env, F>, grad1: Option<Tensor<'env, F>>, t2: Tensor<'env, F>, grad2: Option<Tensor<'env, F>>) -> Option<Tensor<'env, F>> {
        let term1 = grad1.map(|grad1| grad1._mult_with_scalar(t2.r()));
        let term2 = grad2.map(|grad2| t1._mult_with_scalar(grad2));

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


#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_play_around() {
        let env = TensorEnv::new();

        let a = env.create_scalar(2.0);
        let b = env.create_scalar(3.0);
        let c = env.create_scalar(5.0);

        let f = a._mult_with_scalar(b);
        let g = f._plus(c);

        println!("{:?}", g);
    }

    #[test]
    fn test_gradient() {
        let env = TensorEnv::new();

        let a = env.create_scalar(1.0);
        let b = env.create_scalar(3.0);
        let c = env.create_scalar(5.0);

        let x = env.create_scalar(2.0);

        let tracker = RegularExecutionTracker::new();
        // let y = a.plus(x.r(), &tracker);
        let y = a.plus(
            b.mult_element_wise(x.r(), &tracker).plus(
                c.mult_element_wise(
                    x.r().pow_int(2, &tracker),
                    &tracker,
                ),
                &tracker
            ),
            &tracker
        );
        // a + b*x + c*x^2   @ a=1, b=3, c=5, x=2
        println!("{:?}", y);
        println!("dy/da: {:?}", tracker.grad(y.r(), a.r()));
        println!("dy/db: {:?}", tracker.grad(y.r(), b.r()));
        println!("dy/dc: {:?}", tracker.grad(y.r(), c.r()));
    }
}





