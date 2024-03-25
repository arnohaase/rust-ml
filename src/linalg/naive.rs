use std::any::Any;
use std::fmt::{Debug, Display};
use std::ops::{Add, Div, Mul, Sub};


use crate::linalg::{LinAlgFactory, LinAlgMatrix, LinAlgVector, Matrix, Vector};

/// This is the implementation's public API to applications
pub struct LinAlg {}

impl LinAlgFactory for LinAlg {
    // Add<Output=T>+Div<Output=T>+Mul<Output=T>+Sub<Output=T>

    fn initialized_matrix<T: Add<Output=T>+Div<Output=T>+Mul<Output=T>+Sub<Output=T> + Copy + Debug + Display + Default + 'static>(num_rows: usize, num_cols: usize, f: impl Fn(usize, usize) -> T) -> Matrix<T> {
        Matrix::from_raw(Box::new(RawMatrix::new(num_rows, num_cols, f)))
    }

    fn initialized_vector<T: Add<Output=T>+Div<Output=T>+Mul<Output=T>+Sub<Output=T> + Copy + Debug + Display + Default + 'static>(dim: usize, f: impl Fn(usize) -> T) -> Vector<T> {
        Vector::from_raw(Box::new(RawVector::new(dim, f)))
    }
}

#[derive(Clone, Debug)]
struct RawMatrix<T: Add<Output=T>+Div<Output=T>+Mul<Output=T>+Sub<Output=T> + Copy + Debug + Display + Default + 'static> {
    num_rows: usize,
    num_cols: usize,
    /// organized by concatenating rows, row 0 followed by col 1 etc. - see Self::calc_index
    coefficients: Vec<T>,
}
impl <T: Add<Output=T>+Div<Output=T>+Mul<Output=T>+Sub<Output=T> + Copy + Debug + Display + Default + 'static> RawMatrix<T> {
    fn new(num_rows: usize, num_cols: usize, f: impl Fn(usize, usize) -> T) -> RawMatrix<T> {
        let mut coefficients = Vec::new();
        for row in 0..num_rows {
            for col in 0..num_cols {
                coefficients.push(f(row, col));
            }
        }
        RawMatrix {
            num_rows,
            num_cols,
            coefficients,
        }
    }

    fn calc_index(&self, row: usize, col: usize) -> usize {
        row * self.num_cols + col
    }
}
impl <T: Add<Output=T>+Div<Output=T>+Mul<Output=T>+Sub<Output=T> + Copy + Debug + Display + Default + 'static> LinAlgMatrix<T> for RawMatrix<T> {
    fn as_any(&self) -> &dyn Any {
        self
    }


    fn num_rows(&self) -> usize {
        self.num_rows
    }
    fn num_cols(&self) -> usize {
        self.num_cols
    }
    fn box_clone(&self) -> Box<dyn LinAlgMatrix<T>> {
        Box::new(self.clone())
    }

    fn get(&self, row: usize, col: usize) -> T {
        self.coefficients[self.calc_index(row, col)]
    }

    fn set(&mut self, row: usize, col: usize, new_value: T) {
        let index = self.calc_index(row, col);
        self.coefficients[index] = new_value
    }

    fn mult_with_vector(&self, rhs: &dyn LinAlgVector<T>) -> Box<dyn LinAlgVector<T>> {
        let mut result = RawVector::new_zero(self.num_rows);

        for row in 0..self.num_rows {
            for col in 0..self.num_cols {
                result.set(row, result.get(row) + self.get(row, col) * rhs.get(col));
            }
        }

        Box::new(result)
    }

    fn mult_with_scalar(&self, factor: T) -> Box<dyn LinAlgMatrix<T>> {
        let mut result = self.clone();
        for i in 0..result.coefficients.len() {
            let new_value = result.coefficients[i] * factor;
            result.coefficients[i] = new_value;
        }
        Box::new(result)
    }

    fn plus_matrix(&self, rhs: &dyn LinAlgMatrix<T>) -> Box<dyn LinAlgMatrix<T>> {
        let mut result = self.clone();
        result.plus_matrix_in_place(rhs);
        Box::new(result)
    }

    fn minus_matrix(&self, rhs: &dyn LinAlgMatrix<T>) -> Box<dyn LinAlgMatrix<T>> {
        let mut result = self.clone();
        result.minus_matrix_in_place(rhs);
        Box::new(result)
    }

    fn plus_matrix_in_place(&mut self, rhs: &dyn LinAlgMatrix<T>) {
        if let Some(naive) = rhs.as_any().downcast_ref::<RawMatrix<T>>() {
            for i in 0..self.coefficients.len() {
                self.coefficients[i] = self.coefficients[i] + naive.coefficients[i];
            }
        }
        else {
            for row in 0..self.num_rows {
                for col in 0..self.num_cols {
                    let new_value = self.get(row, col) + rhs.get(row, col);
                    self.set(row, col, new_value);
                }
            }
        }
    }

    fn minus_matrix_in_place(&mut self, rhs: &dyn LinAlgMatrix<T>) {
        if let Some(naive) = rhs.as_any().downcast_ref::<RawMatrix<T>>() {
            for i in 0..self.coefficients.len() {
                self.coefficients[i] = self.coefficients[i] - naive.coefficients[i];
            }
        }
        else {
            for row in 0..self.num_rows {
                for col in 0..self.num_cols {
                    let new_value = self.get(row, col) - rhs.get(row, col);
                    self.set(row, col, new_value);
                }
            }
        }
    }

    fn transposed_times_vector(&self, rhs: &dyn LinAlgVector<T>) -> Box<dyn LinAlgVector<T>> {
        assert_eq!(self.num_rows, rhs.dim());

        let mut result = RawVector::new_zero(self.num_cols);

        if let Some(naive) = rhs.as_any().downcast_ref::<RawVector<T>>() {
            for row in 0..self.num_rows {
                for col in 0..self.num_cols {
                    let new_value = result.get(col) + self.get(row, col)*naive.get(row);
                    result.set(col, new_value);
                }
            }
        }
        else {
            for row in 0..self.num_rows {
                for col in 0..self.num_cols {
                    let new_value = result.get(col) + self.get(row, col)*rhs.get(row);
                    result.set(col, new_value);
                }
            }
        }

        Box::new(result)
    }
}

#[derive(Clone, Debug)]
struct RawVector<T: Add<Output=T>+Div<Output=T>+Mul<Output=T>+Sub<Output=T> + Copy + Debug + Display + Default + 'static> {
    values: Vec<T>,
}
impl <T: Add<Output=T>+Div<Output=T>+Mul<Output=T>+Sub<Output=T> + Copy + Debug + Display + Default + 'static> RawVector<T> {
    fn new(dim: usize, f: impl Fn(usize) -> T) -> RawVector<T> {
        let mut values = Vec::new();
        for i in 0..dim {
            values.push(f(i));
        }
        RawVector {
            values,
        }
    }
    fn new_zero(dim: usize) -> RawVector<T> {
        Self::new(dim, |_| Default::default())
    }
}
impl <T: Add<Output=T>+Div<Output=T>+Mul<Output=T>+Sub<Output=T> + Copy + Debug + Display + Default + 'static> LinAlgVector<T> for RawVector<T> {
    fn as_any(&self) -> &dyn Any {
        self
    }


    fn dim(&self) -> usize {
        self.values.len()
    }

    fn box_clone(&self) -> Box<dyn LinAlgVector<T>> {
        Box::new(self.clone())
    }

    fn get(&self, index: usize) -> T {
        self.values[index]
    }

    fn set(&mut self, index: usize, new_value: T) {
        self.values[index] = new_value
    }

    fn multiply_element_wise(&self, rhs: &dyn LinAlgVector<T>) -> Box<dyn LinAlgVector<T>> {
        assert_eq!(self.dim(), rhs.dim());
        let mut result = self.clone();

        if let Some(naive) = rhs.as_any().downcast_ref::<RawVector<T>>() {
            for i in 0..result.dim() {
                let new_value = result.get(i) * naive.get(i);
                result.set(i, new_value);
            }
        }
        else {
            for i in 0..result.dim() {
                let new_value = result.get(i) * rhs.get(i);
                result.set(i, new_value);
            }
        }
        Box::new(result)
    }

    fn map(&self, f: &dyn Fn(T) -> T) -> Box<dyn LinAlgVector<T>> {
        let mut result = self.clone();
        for i in 0..result.dim() {
            let new_value = f(result.get(i));
            result.set(i, new_value);
        }
        Box::new(result)
    }

    fn mult_with_scalar(&self, scalar: T) -> Box<dyn LinAlgVector<T>> {
        let mut result = self.clone();
        for i in 0..self.values.len() {
            let new_value = result.get(i) * scalar;
            result.set(i, new_value);
        }
        Box::new(result)
    }

    fn add_vec(&self, rhs: &dyn LinAlgVector<T>) -> Box<dyn LinAlgVector<T>> {
        assert_eq!(self.dim(), rhs.dim());
        let mut result = self.clone();

        if let Some(naive) = rhs.as_any().downcast_ref::<RawVector<T>>() {
            for i in 0..result.dim() {
                let new_value = result.get(i) + naive.get(i);
                result.set(i, new_value);
            }
        }
        else {
            for i in 0..result.dim() {
                let new_value = result.get(i) + rhs.get(i);
                result.set(i, new_value);
            }
        }
        Box::new(result)
    }

    fn minus_vec_in_place(&mut self, rhs: &dyn LinAlgVector<T>) {
        if let Some(naive) = rhs.as_any().downcast_ref::<RawVector<T>>() {
            for i in 0..self.values.len() {
                self.values[i] = self.values[i] - naive.values[i];
            }
        }
        else {
            for ind in 0..self.values.len() {
                let new_value = self.get(ind) - rhs.get(ind);
                self.set(ind, new_value);
            }
        }
    }


    //TODO move to Matrix
    fn mult_with_transposed_vec(&self, rhs: &dyn LinAlgVector<T>) -> Box<dyn LinAlgMatrix<T>> {
        let mut coefficients = Vec::new();
        let num_rows = self.dim();
        let num_cols;
        if let Some(naive) = rhs.as_any().downcast_ref::<RawVector<T>>() {
            num_cols = naive.dim();
            for row in 0..num_rows {
                for col in 0..num_cols {
                    coefficients.push(self.get(row) * naive.get(col));
                }
            }
        }
        else {
            num_cols = rhs.dim();
            for row in 0..num_rows {
                for col in 0..num_cols {
                    coefficients.push(self.get(row) * rhs.get(col));
                }
            }
        }

        Box::new(RawMatrix {
            num_rows,
            num_cols,
            coefficients,
        })
    }
}
