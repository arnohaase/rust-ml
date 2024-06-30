use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::ops::Deref;
use std::sync::RwLock;

use rustc_hash::FxHashMap;

use crate::n::tensor::Tensor;

pub trait UnaryTensorOp {
    fn calc(&self, tensor: &Tensor) -> Tensor;
    fn grad(&self, t: &Tensor, t_grad: &Option<Tensor>) -> Option<Tensor>;
}
pub trait BinaryTensorOp {
    fn calc(&self, lhs: &Tensor, rhs: &Tensor) -> Tensor;
    fn grad(&self, lhs: &Tensor, lhs_grad: &Option<Tensor>, rhs: &Tensor, rhs_grad: &Option<Tensor>) -> Option<Tensor>;
}

pub enum TrackerExpression {
    Unary(Tensor, Box<dyn UnaryTensorOp>),
    Binary(Tensor, Tensor, Box<dyn BinaryTensorOp>),
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
        result.grad_cache.insert(for_ultimate_input.id(), Some(Tensor::one())); // pre-filling for termination
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

                let cur_result = match expr {
                    TrackerExpression::Unary(t, op) => self.grad_unary(cur, t, op.as_ref()),
                    TrackerExpression::Binary(t1, t2, op) => self.grad_binary(cur, &t1, &t2, op.as_ref()),
                };

                if let Some(g) = cur_result {
                    g
                }
                else {
                    continue;
                }
            }
            else {
                // some part of the calculation that was not tracked by this tracker
                //TODO logging
                None
            };

            if cur_id == calculated.id() {
                return cur_grad;
            }
            self.grad_cache.insert(cur_id, cur_grad);
        }
        None
    }

    fn grad_unary(&mut self, cur: Tensor, t: &Tensor, op: &dyn UnaryTensorOp) -> Option<Option<Tensor>> {
        // 'Unary' means applying a function f to a (potentially calculated) inner
        //   tensor g, with the constraint that f itself is independent of any tracked
        //   variables. So the chain rule applies: ( f(g(x)) )' = f'(g(x) * g'(x)

        // We start with g'(x) because it may turn out not to depend on the
        //  variables we are interested in

        Some(if let Some(inner_grad) = self.grad_cache.get(&t.id()) {
            // the inner gradient is calculated already

            if inner_grad.is_some() {
                op.grad(t, inner_grad)
            } else {
                None
            }
        } else {
            self.work_list.push(cur);
            self.work_list.push(t.clone());
            return None;
        })
    }

    fn grad_binary(&mut self, cur: Tensor, t1: &Tensor, t2: &Tensor, op: &dyn BinaryTensorOp) -> Option<Option<Tensor>> {
        let opt_grad_1 = self.grad_cache.get(&t1.id());
        let opt_grad_2 = self.grad_cache.get(&t2.id());
        Some(match (opt_grad_1, opt_grad_2) {
            (Some(grad_1), Some(grad_2)) => {
                op.grad(t1, grad_1, t2, grad_2)
            }
            _ => {
                self.work_list.push(cur);
                if opt_grad_1.is_some() {
                    self.work_list.push(t1.clone());
                }
                if opt_grad_2.is_some() {
                    self.work_list.push(t2.clone());
                }
                return None;
            }
        })
    }
}