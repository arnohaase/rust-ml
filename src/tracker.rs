use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::ops::Deref;
use std::sync::RwLock;

use rustc_hash::FxHashMap;

use crate::tensor::Tensor;

pub trait UnaryTensorOp: Debug {
    fn calc(&self, tensor: &Tensor) -> Tensor;
    fn grad(&self, t: &Tensor, t_grad: &Option<Tensor>) -> Option<Tensor>;
}
pub trait BinaryTensorOp: Debug {
    fn calc(&self, lhs: &Tensor, rhs: &Tensor) -> Tensor;
    fn grad(&self, lhs: &Tensor, lhs_grad: &Option<Tensor>, rhs: &Tensor, rhs_grad: &Option<Tensor>) -> Option<Tensor>;
}

pub enum TrackerExpression {
    Unary(Tensor, Box<dyn UnaryTensorOp>),
    Binary(Tensor, Tensor, Box<dyn BinaryTensorOp>),
}
impl Debug for TrackerExpression {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            TrackerExpression::Unary(_, op) => write!(f, "{:?}", op),
            TrackerExpression::Binary(_, _, op) => write!(f, "{:?}", op),
        }
    }
}
impl TrackerExpression {
    pub fn calc(&self) -> Tensor {
        match self {
            TrackerExpression::Unary(t, op) => op.calc(t),
            TrackerExpression::Binary(t1, t2, op) => op.calc(t1, t2),
        }
    }
}


pub trait ExecutionTracker {
    fn calc(&self, expr: TrackerExpression) -> Tensor;

    fn grad(&self, calculated: &Tensor, for_ultimate_input: &Tensor) -> Option<Tensor>;
}


pub struct RegularExecutionTracker {
    dependencies: RwLock<HashMap<u32, (u32, TrackerExpression)>>,
}
impl RegularExecutionTracker {
    pub fn new() -> RegularExecutionTracker {
        RegularExecutionTracker {
            dependencies: Default::default()
        }
    }
}
impl ExecutionTracker for RegularExecutionTracker {
    fn calc(&self, expr: TrackerExpression) -> Tensor {
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

    fn grad(&self, calculated: &Tensor, for_ultimate_input: &Tensor) -> Option<Tensor> {
        let mut worker = GradientCalcWorker::new(for_ultimate_input);
        worker.grad(calculated, self.dependencies.read().unwrap())
    }
}

struct GradientCalcWorker {
    grad_cache: FxHashMap<u32, Option<Tensor>>,
    work_list: Vec<Tensor>,
}
impl GradientCalcWorker {
    fn new(for_ultimate_input: &Tensor) -> GradientCalcWorker {
        let mut result = GradientCalcWorker {
            grad_cache: Default::default(),
            work_list: vec![],
        };
        //TODO separate storage to avoid multiplying with 1?
        result.grad_cache.insert(for_ultimate_input.id(), Some(Tensor::scalar(1.0))); // pre-filling for termination
        result
    }

    fn grad(&mut self, calculated: &Tensor, dependencies: impl Deref<Target = HashMap<u32, (u32, TrackerExpression)>>) -> Option<Tensor> {
        self.work_list.push(calculated.clone());

        while let Some(cur) = self.work_list.pop() {
            let cur_id = cur.id();

            let cur_grad: Option<Tensor> = if let Some((version, expr)) = dependencies.get(&cur.id()) {
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

    fn grad_unary(&mut self, cur: &Tensor, t: &Tensor, op: &dyn UnaryTensorOp) -> Option<Option<Tensor>> {
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

    fn grad_binary(&mut self, cur: &Tensor, t1: &Tensor, t2: &Tensor, op: &dyn BinaryTensorOp) -> Option<Option<Tensor>> {
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
    use crate::binop_minus::BinOpMinus;
    use crate::binop_mult::BinOpMult;
    use crate::binop_plus::BinOpPlus;
    use crate::binop_polynomial::BinOpPolynomial;

    use crate::tensor::{DimensionKind, Tensor};
    use crate::tracker::{ExecutionTracker, RegularExecutionTracker, TrackerExpression};
    use crate::unop_avg::UnOpAvg;

    #[test]
    fn test_sin_poly_mult() {
        const EPS: f64 = 1e-2;

        // approximate sin(x) by a*x^3 + b*x^2 + c*x + d

        let mut a = Tensor::scalar(rand::thread_rng().gen_range(-1.0..1.0));
        let mut b = Tensor::scalar(rand::thread_rng().gen_range(-1.0..1.0));
        let mut c = Tensor::scalar(rand::thread_rng().gen_range(-1.0..1.0));
        let mut d = Tensor::scalar(rand::thread_rng().gen_range(-1.0..1.0));

        let mut xs = Vec::new();
        let mut y_ref = Vec::new();
        for _ in 0..2000 {
            let x: f64 = rand::thread_rng().gen_range(-1.6..1.6);
            xs.push(x);
            y_ref.push(x.sin());
        }
        let xs = Tensor::vector(xs, DimensionKind::Collection);
        let y_ref = Tensor::vector(y_ref, DimensionKind::Collection);

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
        const EPS: f64 = 1e-2;

        // approximate sin(x) by a*x^3 + b*x^2 + c*x + d

        let mut poly = Tensor::vector(vec![
            rand::thread_rng().gen_range(-1.0..1.0),
            rand::thread_rng().gen_range(-1.0..1.0),
            rand::thread_rng().gen_range(-1.0..1.0),
            rand::thread_rng().gen_range(-1.0..1.0),
        ], DimensionKind::Polynomial);

        let mut xs = Vec::new();
        let mut y_ref = Vec::new();
        for _ in 0..2000 {
            let x: f64 = rand::thread_rng().gen_range(-1.6..1.6);
            xs.push(x);
            y_ref.push(x.sin());
        }
        let xs = Tensor::vector(xs, DimensionKind::Collection);
        let y_ref = Tensor::vector(y_ref, DimensionKind::Collection);

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
