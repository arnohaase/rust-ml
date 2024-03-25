use std::cell::RefCell;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::rc::Rc;
use std::time::Instant;

use crate::layer::{Dense, Layer, Tanh};
use crate::linalg::naive::LinAlg;
use crate::linalg::*;

pub mod linalg;
pub mod layer;

fn main() -> std::io::Result<()>{
    println!("starting at {:?}", Instant::now());

    let (x_train, y_train, x_test, y_test) = read_mnist()?;
    assert_eq!(x_train.len(), y_train.len());
    assert_eq!(x_test.len(), y_test.len());

    let network: Vec<Rc<RefCell<dyn Layer>>> = vec![
        Rc::new(RefCell::new(Dense::new(28*28, 28*28))),
        Rc::new(RefCell::new(Tanh::new())),
        Rc::new(RefCell::new(Dense::new(28*28, 28*28))),
        Rc::new(RefCell::new(Tanh::new())),
        Rc::new(RefCell::new(Dense::new(28*28, 40))),
        Rc::new(RefCell::new(Tanh::new())),
        Rc::new(RefCell::new(Dense::new(40, 10))),
        Rc::new(RefCell::new(Tanh::new())),
    ];

    println!("start training at {:?}", Instant::now());

    train(&network, mse, mse_prime, &x_train, &y_train, 400, 0.001, true);

    let mut num_predicted_correctly = 0;
    for ind in 0..x_test.len() {
        let output = predict(&network, &x_test[ind]);
        let predicted = arg_max(&output);
        let annotated = arg_max(&y_test[ind]);
        if predicted == annotated {
            num_predicted_correctly += 1;
        }
    }
    println!("test data: {num_predicted_correctly} / {} predicted correctly: {}%", y_test.len(), (num_predicted_correctly / y_test.len())*100);

    Ok(())
}

fn read_mnist() -> std::io::Result<(Vec<Vector<f64>>,Vec<Vector<f64>>,Vec<Vector<f64>>,Vec<Vector<f64>>,)> {
    fn read_file(path: &str) -> std::io::Result<(Vec<Vector<f64>>, Vec<Vector<f64>>)> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let mut all_x= Vec::new();
        let mut all_y= Vec::new();

        for line in reader.lines() {
            let line = line?;

            let parts: Vec<&str> = line.split(',').collect();

            let correct_digit = parts[0];
            let image_data = &parts[1..];
            assert_eq!(28*28, image_data.len());

            let x = LinAlg::initialized_vector(28*28, |ind| (image_data[ind].parse::<u8>().unwrap() as f64) / 255.0);
            all_x.push(x);

            let correct_digit = correct_digit.parse::<usize>().unwrap();
            let y = LinAlg::initialized_vector(10, |i| if i == correct_digit { 1.0 } else {0.0 });
            all_y.push(y);
        }

        Ok((all_x, all_y))
    }

    println!("reading training data");
    let (x_train, y_train) = read_file("data/mnist/mnist_train.csv")?;
    println!("reading test data");
    let (x_test, y_test) = read_file("data/mnist/mnist_test.csv")?;
    Ok((x_train, y_train, x_test, y_test))
}

fn arg_max(v: &Vector<f64>) -> usize {
    assert!(v.dim() > 0);
    let mut max_index = 0;
    let mut max_value = v.get(0);

    for i in 1..v.dim() {
        let cur_value = v.get(i);
        if cur_value > max_value {
            max_index = i;
            max_value = cur_value;
        }
    }
    max_index
}


fn mse(actual: &Vector<f64>, predicted: &Vector<f64>) -> f64 {
    assert_eq!(actual.dim(), predicted.dim());
    let mut result = 0.0;
    for i in 0..actual.dim() {
        let diff = actual.get(i) - predicted.get(i);
        result += diff*diff;
    }
    result * (1.0 / (actual.dim() as f64))
}

fn mse_prime(actual: &Vector<f64>, predicted: &Vector<f64>) -> Vector<f64> {
    assert_eq!(actual.dim(), predicted.dim());

    let mut result = LinAlg::zero_vector(actual.dim());
    for i in 0..actual.dim() {
        result.set(i, 2.0 * (predicted.get(i) - actual.get(i)) / (actual.dim() as f64)); //TODO why divided by the dimension?
    }
    result
}

fn predict(network: &[Rc<RefCell<dyn Layer>>], input: &Vector<f64>) -> Vector<f64> {
    let mut output = input.clone(); //TODO clone?!
    for layer in network {
        output = layer.borrow().forward(output);
    }
    output
}

fn train(
    network: &[Rc<RefCell<dyn Layer>>],
    loss: impl Fn(&Vector<f64>, &Vector<f64>) -> f64,
    loss_prime: impl Fn(&Vector<f64>, &Vector<f64>) -> Vector<f64>,
    x_train: &Vec<Vector<f64>>,
    y_train: &Vec<Vector<f64>>,
    epochs: usize,
    learning_rate: f64,
    verbose: bool,
) {
    for e in 0..epochs {
        let mut error = 0.0;

        //TODO 500?
        for ind in 0..50 { //x_train.len() {
            let x = &x_train[ind];
            let y = &y_train[ind];

            let output = predict(network, x);
            error += loss(y, &output);

            let mut grad = loss_prime(y, &output);
            for layer in network.iter().rev() {
                grad = layer.borrow_mut().backward(grad, learning_rate);
            }
        }

        error /= 50.0;
        // error /= x_train.len() as f64;
        if verbose {
            println!("{:?}: epoch {e}/{epochs}: error {error}", Instant::now());
        }
    }
}
