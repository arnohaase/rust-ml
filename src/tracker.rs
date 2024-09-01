use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::ops::Deref;
use std::sync::RwLock;

use rustc_hash::FxHashMap;

use crate::tensor::Tensor;
use crate::tensor_env::TensorEnv;

pub trait UnaryTensorOp<E: TensorEnv>: Debug {
    fn calc<'env>(&self, tensor: &Tensor<'env, E>) -> Tensor<'env, E>;
    fn grad<'env>(&self, t: &Tensor<'env, E>, t_grad: &Option<Tensor<'env, E>>) -> Option<Tensor<'env, E>>;
}
pub trait BinaryTensorOp<E: TensorEnv>: Debug {
    fn calc<'env>(&self, lhs: &Tensor<'env, E>, rhs: &Tensor<'env, E>) -> Tensor<'env, E>;
    fn grad<'env>(&self, lhs: &Tensor<'env, E>, lhs_grad: &Option<Tensor<'env, E>>, rhs: &Tensor<'env, E>, rhs_grad: &Option<Tensor<'env, E>>) -> Option<Tensor<'env, E>>;
}

pub enum TrackerExpression<'env, E: TensorEnv> {
    Unary(Tensor<'env, E>, Box<dyn UnaryTensorOp<E>>),
    Binary(Tensor<'env, E>, Tensor<'env, E>, Box<dyn BinaryTensorOp<E>>),
}
impl <'env, E: TensorEnv> Debug for TrackerExpression<'env, E> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            TrackerExpression::Unary(_, op) => write!(f, "{:?}", op),
            TrackerExpression::Binary(_, _, op) => write!(f, "{:?}", op),
        }
    }
}
impl <'env, E: TensorEnv> TrackerExpression<'env, E> {
    pub fn calc(&self) -> Tensor<'env, E> {
        match self {
            TrackerExpression::Unary(t, op) => op.calc(t),
            TrackerExpression::Binary(t1, t2, op) => op.calc(t1, t2),
        }
    }
}


pub trait ExecutionTracker<'env, E: TensorEnv> {
    fn calc(&self, expr: TrackerExpression<'env, E>) -> Tensor<'env, E>;

    fn grad(&self, calculated: &Tensor<'env, E>, for_ultimate_input: &Tensor<'env, E>) -> Option<Tensor<'env, E>>;
}


pub struct RegularExecutionTracker<'env, E: TensorEnv> {
    dependencies: RwLock<HashMap<u32, (u32, TrackerExpression<'env, E>)>>,
}
impl <'env, E: TensorEnv> RegularExecutionTracker<'env, E> {
    pub fn new() -> RegularExecutionTracker<'env, E> {
        RegularExecutionTracker {
            dependencies: Default::default()
        }
    }
}
impl <'env, E: TensorEnv> ExecutionTracker<'env, E> for RegularExecutionTracker<'env, E> {
    fn calc(&self, expr: TrackerExpression<'env, E>) -> Tensor<'env, E> {
        let calculated = expr.calc();
        let version = calculated.version();
        let mut write = self.dependencies.write().unwrap();
        match write.entry(calculated.id()) {
            Entry::Occupied(_) => {
                eprintln!("TODO - cyclic"); //TODO error handling, error reporting
            }
            Entry::Vacant(e) => {
                e.insert((version, expr));
            }
        }
        calculated
    }

    fn grad(&self, calculated: &Tensor<'env, E>, for_ultimate_input: &Tensor<'env, E>) -> Option<Tensor<'env, E>> {
        let mut worker = GradientCalcWorker::new(for_ultimate_input);
        worker.grad(calculated, self.dependencies.read().unwrap())
    }
}

struct GradientCalcWorker<'env, E: TensorEnv> {
    grad_cache: FxHashMap<u32, Option<Tensor<'env, E>>>,
    work_list: Vec<Tensor<'env, E>>,
}
impl <'env, E: TensorEnv> GradientCalcWorker<'env, E> {
    fn new(for_ultimate_input: &Tensor<'env, E>) -> GradientCalcWorker<'env, E> {
        let mut result = GradientCalcWorker {
            grad_cache: Default::default(),
            work_list: vec![],
        };
        //TODO separate storage to avoid multiplying with 1 --> optimization
        result.grad_cache.insert(for_ultimate_input.id(), Some(for_ultimate_input.env().scalar(1.0))); // pre-filling for termination
        result
    }

    fn grad(&mut self, calculated: &Tensor<'env, E>, dependencies: impl Deref<Target = HashMap<u32, (u32, TrackerExpression<'env, E>)>>) -> Option<Tensor<'env, E>> {
        self.work_list.push(calculated.clone());

        while let Some(cur) = self.work_list.pop() {
            let cur_id = cur.id();

            let cur_grad: Option<Tensor<'env, E>> = if let Some((version, expr)) = dependencies.get(&cur.id()) {
                if *version != cur.version() {
                    // the calculated tensor was modified in place since the gradient was calculated
                    todo!()
                }

                // println!("  {:?} -> {:?}: {:?}",
                //          self.work_list.iter().map(|i| i.id()).collect::<Vec<_>>(),
                //          cur_id,
                //          expr);

                let cur_result = match expr {
                    TrackerExpression::Unary(t, op) => self.grad_unary(&cur, t, op.as_ref()),
                    TrackerExpression::Binary(t1, t2, op) => self.grad_binary(&cur, &t1, &t2, op.as_ref()),
                };

                if let Some(g) = cur_result {
                    // println!("    {:?}: {:?} {:?} -> {:?}",
                    //     self.work_list.iter().map(|i| i.id()).collect::<Vec<_>>(),
                    //     cur_id,
                    //     expr,
                    //     g,
                    // );
                    //
                    g
                }
                else {
                    // println!("    ...");
                    continue;
                }
            }
            else {
                // some part of the calculation that was not tracked by this tracker, e.g. literals
                None
            };

            if cur_id == calculated.id() {
                return cur_grad;
            }
            self.grad_cache.insert(cur_id, cur_grad);
        }
        None
    }

    fn grad_unary(&mut self, cur: &Tensor<'env, E>, t: &Tensor<'env, E>, op: &dyn UnaryTensorOp<E>) -> Option<Option<Tensor<'env, E>>> {
        // 'Unary' means applying a function f to a (potentially calculated) inner
        //   tensor g, with the constraint that f itself is independent of any tracked
        //   variables. So the chain rule applies: ( f(g(x)) )' = f'(g(x) * g'(x)

        // We start with g'(x) because it may turn out not to depend on the
        //  variables we are interested in

        if let Some(inner_grad) = self.grad_cache.get(&t.id()) {
            // the inner gradient is calculated already

            if inner_grad.is_some() {
                Some(op.grad(t, inner_grad))
            }
            else {
                Some(None)
            }
        }
        else {
            self.work_list.push(cur.clone());
            self.work_list.push(t.clone()); //TODO store ids in the expressions?
            None
        }
    }

    fn grad_binary(&mut self, cur: &Tensor<'env, E>, t1: &Tensor<'env, E>, t2: &Tensor<'env, E>, op: &dyn BinaryTensorOp<E>) -> Option<Option<Tensor<'env, E>>> {
        let opt_grad_1 = self.grad_cache.get(&t1.id());
        let opt_grad_2 = self.grad_cache.get(&t2.id());
        match (opt_grad_1, opt_grad_2) {
            (Some(grad_1), Some(grad_2)) =>
                Some(op.grad(t1, grad_1, t2, grad_2)),
            _ => {
                self.work_list.push(cur.clone());
                if opt_grad_1.is_none() {
                    self.work_list.push(t1.clone());
                }
                if opt_grad_2.is_none() {
                    self.work_list.push(t2.clone());
                }
                None
            }
        }
    }
}

#[cfg(test)]
mod test {
    use rand::Rng;
    use crate::dimension::DimensionKind;
    use crate::operations::binop_minus::BinOpMinus;
    use crate::operations::binop_mult::BinOpMult;
    use crate::operations::binop_plus::BinOpPlus;
    use crate::operations::binop_polynomial::BinOpPolynomial;

    use crate::tensor_env::{BlasEnv, TensorEnv};
    use crate::tracker::{ExecutionTracker, RegularExecutionTracker, TrackerExpression};
    use crate::operations::unop_avg::UnOpAvg;

    #[test]
    fn test_sin_poly_mult() {
        const EPS: f32 = 1e-2;

        let env = BlasEnv{};

        // approximate sin(x) by a*x^3 + b*x^2 + c*x + d

        let mut a = env.scalar(rand::thread_rng().gen_range(-1.0..1.0));
        let mut b = env.scalar(rand::thread_rng().gen_range(-1.0..1.0));
        let mut c = env.scalar(rand::thread_rng().gen_range(-1.0..1.0));
        let mut d = env.scalar(rand::thread_rng().gen_range(-1.0..1.0));

        let mut xs = Vec::new();
        let mut y_ref = Vec::new();
        for _ in 0..2000 {
            let x: f32 = rand::thread_rng().gen_range(-1.6..1.6);
            xs.push(x);
            y_ref.push(x.sin());
        }
        let xs = env.vector(xs, DimensionKind::Collection);
        let y_ref = env.vector(y_ref, DimensionKind::Collection);

        for n in 0..10_000 {
            let tracker = RegularExecutionTracker::new();

            let t3 = tracker.calc(TrackerExpression::Binary(xs.clone(), xs.clone(), Box::new(BinOpMult {})));
            let t3 = tracker.calc(TrackerExpression::Binary(t3.clone(), xs.clone(), Box::new(BinOpMult {})));
            let t3 = tracker.calc(TrackerExpression::Binary(t3.clone(), a.clone(), Box::new(BinOpMult {})));

            let t2 = tracker.calc(TrackerExpression::Binary(xs.clone(), xs.clone(), Box::new(BinOpMult {})));
            let t2 = tracker.calc(TrackerExpression::Binary(t2.clone(), b.clone(), Box::new(BinOpMult {})));

            let t1 = tracker.calc(TrackerExpression::Binary(xs.clone(), c.clone(), Box::new(BinOpMult {})));

            let poly = tracker.calc(TrackerExpression::Binary(t3, t2, Box::new(BinOpPlus {})));
            let poly = tracker.calc(TrackerExpression::Binary(poly, t1, Box::new(BinOpPlus {})));
            let poly = tracker.calc(TrackerExpression::Binary(poly, d.clone(), Box::new(BinOpPlus {})));

            let dy = tracker.calc(TrackerExpression::Binary(poly, y_ref.clone(), Box::new(BinOpMinus {})));
            let dy = tracker.calc(TrackerExpression::Binary(dy.clone(), dy, Box::new(BinOpMult {})));

            let err = tracker.calc(TrackerExpression::Unary(dy, Box::new(UnOpAvg {})));

            let grad_a = tracker.grad(&err, &a).unwrap();
            let grad_b = tracker.grad(&err, &b).unwrap();
            let grad_c = tracker.grad(&err, &c).unwrap();
            let grad_d = tracker.grad(&err, &d).unwrap();

            BinOpPlus::plus_in_place(&mut a, &grad_a, -EPS);
            BinOpPlus::plus_in_place(&mut b, &grad_b, -EPS);
            BinOpPlus::plus_in_place(&mut c, &grad_c, -EPS);
            BinOpPlus::plus_in_place(&mut d, &grad_d, -EPS);

            if n%100 == 0 {
                println!("{n}: {:?}", err);
            }

            if err.buf().read().unwrap()[0] < 1e-5 {
                break;
            }
        }
    }

    #[test]
    fn test_sin_poly_builtin() {
        const EPS: f32 = 1e-2;

        let env = BlasEnv{};

        // approximate sin(x) by a*x^3 + b*x^2 + c*x + d

        let mut poly = env.vector(vec![
            rand::thread_rng().gen_range(-1.0..1.0),
            rand::thread_rng().gen_range(-1.0..1.0),
            rand::thread_rng().gen_range(-1.0..1.0),
            rand::thread_rng().gen_range(-1.0..1.0),
        ], DimensionKind::Polynomial);

        let mut xs = Vec::new();
        let mut y_ref = Vec::new();
        for _ in 0..2000 {
            let x: f32 = rand::thread_rng().gen_range(-1.6..1.6);
            xs.push(x);
            y_ref.push(x.sin());
        }
        let xs = env.vector(xs, DimensionKind::Collection);
        let y_ref = env.vector(y_ref, DimensionKind::Collection);

        for n in 0..10_000 {
            let tracker = RegularExecutionTracker::new();

            let p = tracker.calc(TrackerExpression::Binary(poly.clone(), xs.clone(), Box::new(BinOpPolynomial{})));

            let dy = tracker.calc(TrackerExpression::Binary(p, y_ref.clone(), Box::new(BinOpMinus {})));
            let dy = tracker.calc(TrackerExpression::Binary(dy.clone(), dy, Box::new(BinOpMult {})));

            let err = tracker.calc(TrackerExpression::Unary(dy, Box::new(UnOpAvg {})));

            let grad = tracker.grad(&err, &poly).unwrap();
            BinOpPlus::plus_in_place(&mut poly, &grad, -EPS);

            if n%100 == 0 {
                println!("{n}: {:?}", err);
            }
            if err.buf().read().unwrap()[0] < 1e-5 {
                break;
            }
        }
    }
}
