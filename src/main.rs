use std::cell::RefCell;
use std::cmp::Ordering;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::ops::Deref;
use std::rc::Rc;
use std::time::Instant;
use crate::layer::{Dense, Layer, Tanh};
use crate::linalg_initial::Vector;

pub mod linalg_initial;
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

    train(&network, mse, mse_prime, &x_train, &y_train, 400, 0.001, true);

    let mut num_predicted_correctly = 0;
    for ind in 0..x_test.len() {
        let output = predict(&network, x_test[ind].clone());
        let predicted = arg_max(output.deref());
        let annotated = arg_max(y_test[ind].deref());
        if predicted == annotated {
            num_predicted_correctly += 1;
        }
    }
    println!("test data: {num_predicted_correctly} / {} predicted correctly: {}%", y_test.len(), (num_predicted_correctly / y_test.len())*100);

    Ok(())
}

fn read_mnist() -> std::io::Result<(Vec<Rc<Vector>>,Vec<Rc<Vector>>,Vec<Rc<Vector>>,Vec<Rc<Vector>>,)> {
    fn read_file(path: &str) -> std::io::Result<(Vec<Rc<Vector>>, Vec<Rc<Vector>>)> {
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

            let mut x = Vector::new_zero(28*28);
            for ind in 0..image_data.len() {
                x[ind] = (image_data[ind].parse::<u8>().unwrap() as f64) / 255.0;
            }
            all_x.push(Rc::new(x));

            let mut y = Vector::new_zero(10);
            y[correct_digit.parse::<usize>().unwrap()] = 1.0;
            all_y.push(Rc::new(y));
        }

        Ok((all_x, all_y))
    }

    println!("reading training data");
    let (x_train, y_train) = read_file("data/mnist/mnist_train.csv")?;
    println!("reading test data");
    let (x_test, y_test) = read_file("data/mnist/mnist_test.csv")?;
    Ok((x_train, y_train, x_test, y_test))
}

fn arg_max(v: &Vector) -> usize {
    v.values.iter()
        .enumerate()
        .max_by(|a, b| PartialOrd::partial_cmp(a.1, b.1).unwrap_or(Ordering::Equal))
        .map(|(index, _)| index)
        .unwrap()
}


fn mse(actual: &Vector, predicted: &Vector) -> f64 {
    assert_eq!(actual.dim(), predicted.dim());
    let mut result = 0.0;
    for i in 0..actual.dim() {
        let diff = actual[i] - predicted[i];
        result += diff*diff;
    }
    result * (1.0 / (actual.dim() as f64))
}

fn mse_prime(actual: &Vector, predicted: &Vector) -> Vector {
    assert_eq!(actual.dim(), predicted.dim());

    let mut result = Vector::new_zero(actual.dim());
    for i in 0..actual.dim() {
        result[i] = 2.0 * (predicted[i] - actual[i]) / (actual.dim() as f64); //TODO why divided by the dimension?
    }
    result
}

fn predict(network: &[Rc<RefCell<dyn Layer>>], input: Rc<Vector>) -> Rc<Vector> {
    let mut output = input;
    for layer in network {
        output = Rc::new(layer.borrow().forward(output));
    }
    output
}

fn train(
    network: &[Rc<RefCell<dyn Layer>>],
    loss: impl Fn(&Vector, &Vector) -> f64,
    loss_prime: impl Fn(&Vector, &Vector) -> Vector,
    x_train: &Vec<Rc<Vector>>,
    y_train: &Vec<Rc<Vector>>,
    epochs: usize,
    learning_rate: f64,
    verbose: bool,
) {
    for e in 0..epochs {
        let mut error = 0.0;

        for ind in 0..500 { //x_train.len() {
            let x = &x_train[ind];
            let y = &y_train[ind];

            let output = predict(network, x.clone());
            error += loss(y, output.deref());

            let mut grad = Rc::new(loss_prime(y, output.deref()));
            for layer in network.iter().rev() {
                grad = Rc::new(layer.borrow_mut().backward(grad, learning_rate));
            }
        }

        error /= 500.0;
        // error /= x_train.len() as f64;
        if verbose {
            println!("{:?}: epoch {e}/{epochs}: error {error}", Instant::now());
        }
    }
}
