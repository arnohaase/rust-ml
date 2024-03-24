use std::cell::RefCell;
use std::ops::Deref;
use std::rc::Rc;
use crate::linalg::{Matrix, Vector};

pub trait Layer {
    fn forward(&self, input: Rc<Vector>) -> Vector;

    fn backward(&mut self, output_gradient: Rc<Vector>, learning_rate: f64) -> Vector;
}


pub struct Dense {
    weights: Matrix,
    bias: Vector,
    input: RefCell<Option<Rc<Vector>>>,
}
impl Dense {
    pub fn new(num_in: usize, num_out: usize) -> Dense {
        Dense {
            weights: Matrix::new_random(num_out, num_in),
            bias: Vector::new_random(num_out),
            input: Default::default(),
        }
    }
}

impl Layer for Dense {
    fn forward(&self, input: Rc<Vector>) -> Vector {
        let _ = self.input.borrow_mut().insert(input.clone());
        let a = &self.weights * input.deref();
        a + &self.bias
    }

    fn backward(&mut self, output_gradient: Rc<Vector>, learning_rate: f64) -> Vector {
        let weights_gradient = vec_times_transposed_vec(output_gradient.deref(), self.input.borrow().as_ref().unwrap().as_ref());
        let input_gradient = self.weights.transposed_times(output_gradient.as_ref());

        self.weights -= &(&weights_gradient * learning_rate);
        self.bias -= &(output_gradient.as_ref() * learning_rate);

        input_gradient
    }
}

fn vec_times_transposed_vec(v1: &Vector, v2: &Vector) -> Matrix {
    let mut result = Matrix::new_zero(v1.dim(), v2.dim());

    for row in 0..v1.dim() {
        for col in 0..v2.dim() {
            result.coefficients[row*v1.dim() + col] = v1[row] * v2[col];
        }
    }
    result
}


pub struct Tanh {
    input: RefCell<Option<Rc<Vector>>>,
}
impl Tanh {
    pub fn new() -> Tanh {
        Tanh {
            input: Default::default(),
        }
    }
}

impl Layer for Tanh {
    fn forward(&self, input: Rc<Vector>) -> Vector {
        let _ = self.input.borrow_mut().insert(input.clone());
        let mut result = input.deref().clone();
        for i in 0..result.dim() {
            result[i] = result[i].tanh();
        }
        result
    }

    fn backward(&mut self, output_gradient: Rc<Vector>, _learning_rate: f64) -> Vector {
        let input = self.input.borrow().as_ref().unwrap().clone();
        let mapped = input.as_ref().map(|x| 1.0 - x.tanh()*x.tanh());
        output_gradient.multiply_element_wise(&mapped)
    }
}






