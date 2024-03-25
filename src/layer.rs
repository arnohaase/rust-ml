use std::cell::RefCell;
use rand::random;

use crate::linalg::*;
use crate::linalg::naive::LinAlg;

pub trait Layer {
    fn forward(&self, input: Vector<f64>) -> Vector<f64>;

    fn backward(&mut self, output_gradient: Vector<f64>, learning_rate: f64) -> Vector<f64>;
}


pub struct Dense {
    weights: Matrix<f64>,
    bias: Vector<f64>,
    input: RefCell<Vector<f64>>,
}
impl Dense {
    pub fn new(num_in: usize, num_out: usize) -> Dense {
        Dense {
            weights: LinAlg::initialized_matrix(num_out, num_in, |_,_| random::<f64>() * 2.0 - 1.0), // Matrix<f64>::new_random(num_out, num_in),
            bias: LinAlg::initialized_vector(num_out, |_| random::<f64>() * 2.0 - 1.0), //Vector<f64>::new_random(num_out),
            input: RefCell::new(LinAlg::zero_vector(0)), //TODO
        }
    }
}

impl Layer for Dense {
    fn forward(&self, input: Vector<f64>) -> Vector<f64> {
        let a = &self.weights * &input;
        let _ = self.input.replace(input);
        &a + &self.bias
    }

    fn backward(&mut self, output_gradient: Vector<f64>, learning_rate: f64) -> Vector<f64> {
        let weights_gradient = output_gradient.mult_with_transposed_vec(&self.input.borrow());
        let input_gradient = self.weights.transposed_times_vector(&output_gradient);

        // println!("{:?}", self.bias);
        // println!("- {:?}", output_gradient);

        self.weights -= &(&weights_gradient * learning_rate); //TODO variants that accept ownership
        self.bias -= &(&output_gradient * learning_rate);

        // println!("--> {:?}", self.bias);
        // println!("----");

        input_gradient
    }
}


pub struct Tanh {
    input: RefCell<Vector<f64>>,
}
impl Tanh {
    pub fn new() -> Tanh {
        Tanh {
            input: RefCell::new(LinAlg::zero_vector(0)), //TODO
        }
    }
}

impl Layer for Tanh {
    fn forward(&self, input: Vector<f64>) -> Vector<f64> {
        let _ = self.input.replace(input.clone()); //TODO clone?!
        input.map(|x| x.tanh())
    }

    fn backward(&mut self, output_gradient: Vector<f64>, _learning_rate: f64) -> Vector<f64> {
        let mapped = self.input.borrow().map(|x| 1.0 - x.tanh()*x.tanh());
        output_gradient.multiply_element_wise(&mapped)
    }
}

pub struct ReLu {
    input: RefCell<Vector<f64>>,
}
impl ReLu {
    pub fn new() -> ReLu {
        ReLu {
            input: RefCell::new(LinAlg::zero_vector(0)), //TODO
        }
    }
}

impl Layer for ReLu {
    fn forward(&self, input: Vector<f64>) -> Vector<f64> {
        let _ = self.input.replace(input.clone()); //TODO clone?!
        input.map(|x| if x < 0.0 { 0.0 } else { x })
    }

    fn backward(&mut self, output_gradient: Vector<f64>, _learning_rate: f64) -> Vector<f64> {
        let mapped = self.input.borrow().map(|x| if x < 0.0 { 0.0 } else { 1.0 });
        output_gradient.multiply_element_wise(&mapped)
    }
}






