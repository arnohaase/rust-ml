pub mod naive;

use std::any::Any;
use std::cell::RefCell;
use std::fmt::{Debug, Display};
use std::ops::{Add, Div, Mul, Sub, SubAssign};
use triomphe::Arc;



//TODO Vector / Matrix as enum: Mutable, Immutable, Calculated (with 'materialize()' -> inner Enum that swaps out the result for the program), Shared (?)
//  -> or just regular Rust mut / regular ref?



pub trait LinAlgFactory {
    fn zero_matrix<T: Add<Output=T>+Div<Output=T>+Mul<Output=T>+Sub<Output=T> + Copy + Debug + Display + Default + 'static>(num_rows: usize, num_cols: usize) -> Matrix<T> {
        Self::initialized_matrix(num_rows, num_cols, |_,_| Default::default())
    }
    fn initialized_matrix<T: Add<Output=T>+Div<Output=T>+Mul<Output=T>+Sub<Output=T> + Copy + Debug + Display + Default + 'static>(num_rows: usize, num_cols: usize, f: impl Fn(usize, usize) -> T) -> Matrix<T>;

    //TODO convenience methods for different kinds of 'random' for different types

    fn zero_vector<T: Add<Output=T>+Div<Output=T>+Mul<Output=T>+Sub<Output=T> + Copy + Debug + Display + Default + 'static> (dim: usize) -> Vector<T> {
        Self::initialized_vector(dim, |_| Default::default())
    }
    fn initialized_vector<T: Add<Output=T>+Div<Output=T>+Mul<Output=T>+Sub<Output=T> + Copy + Debug + Display + Default + 'static> (dim: usize, f: impl Fn(usize) -> T) -> Vector<T>;

    //TODO convenience methods for different kinds of 'random' for different types
}


pub struct Matrix<T> {
    raw: Arc<RefCell<Box<dyn LinAlgMatrix<T>>>>,
}

impl <T> Matrix<T> {
    pub fn from_raw(raw: Box<dyn LinAlgMatrix<T>>) -> Matrix<T> {
        Matrix {
            raw: Arc::new(RefCell::new(raw))
        }
    }

    pub fn r(&self) -> Matrix<T> {
        Matrix {
            raw: self.raw.clone(),
        }
    }

    pub fn copy(&self) -> Matrix<T> {
        Matrix {
            raw: Arc::new(RefCell::new(self.raw.borrow().box_clone())),
        }
    }

    pub fn num_rows(&self) -> usize {
        self.raw.borrow().num_rows()
    }
    pub fn num_cols(&self) -> usize {
        self.raw.borrow().num_cols()
    }

    pub fn get(&self, row: usize, col: usize) -> T {
        self.raw.borrow().get(row, col)
    }
    pub fn set(&self, row: usize, col: usize, new_value: T) {
        self.raw.borrow_mut().set(row, col, new_value)
    }

    //TODO 'program' built from 'primitives'
    pub fn transposed_times_vector(&self, rhs: Vector<T>) -> Vector<T> {
        Vector::from_raw(self.raw.borrow().transposed_times_vector(rhs.raw.borrow().as_ref()))
    }
}

impl <T> Mul<Vector<T>> for Matrix<T> {
    type Output = Vector<T>;

    fn mul(self, rhs: Vector<T>) -> Self::Output {
        assert_eq!(self.num_cols(), rhs.dim());
        //TODO lazy / deferred
        Vector::from_raw(self.raw.borrow().mult_with_vector(rhs.raw.borrow().as_ref()))
    }
}

impl <T> Mul<T> for Matrix<T> {
    type Output = Matrix<T>;

    fn mul(self, rhs: T) -> Self::Output {
        //TODO lazy / deferred
        Matrix::from_raw(self.raw.borrow().mult_with_scalar(rhs))
    }
}

impl <T> SubAssign<Matrix<T>> for Matrix<T> {
    fn sub_assign(&mut self, rhs: Matrix<T>) {
        assert_eq!(self.num_rows(), rhs.num_rows());
        assert_eq!(self.num_cols(), rhs.num_cols());

        self.raw.borrow_mut().minus_matrix_in_place(rhs.raw.borrow().as_ref())
    }
}


#[derive(Debug)]
pub struct Vector<T> {
    raw: Arc<RefCell<Box<dyn LinAlgVector<T>>>>,
}
impl <T> Vector<T> {
    fn from_raw(raw: Box<dyn LinAlgVector<T>>) -> Vector<T> {
        Vector {
            raw: Arc::new(RefCell::new(raw))
        }
    }

    pub fn r(&self) -> Vector<T> {
        Vector {
            raw: self.raw.clone(),
        }
    }

    pub fn copy(&self) -> Vector<T> {
        Self::from_raw(self.raw.borrow().box_clone())
   }

    pub fn get(&self, index: usize) -> T {
        self.raw.borrow().get(index)
    }
    pub fn set(&self, index: usize, new_value: T) {
        self.raw.borrow_mut().set(index, new_value)
    }

    pub fn dim(&self) -> usize {
        self.raw.borrow().dim()
    }

    //TODO pass references for non-consumed parameters?
    pub fn mult_with_transposed_vec(&self, rhs: Vector<T>) -> Matrix<T> {
        Matrix::from_raw(self.raw.borrow().mult_with_transposed_vec(rhs.raw.borrow().as_ref()))
    }

    pub fn map(&self, f: impl Fn(T) -> T) -> Vector<T> {
        Vector::from_raw(self.raw.borrow().map(&f))
    }

    pub fn multiply_element_wise(&self, rhs: Vector<T>) -> Vector<T> {
        Vector::from_raw(self.raw.borrow().multiply_element_wise(rhs.raw.borrow().as_ref()))
    }
}

//TODO syntactic sugar for indexed get / set syntax

impl <T> Add<Vector<T>> for Vector<T> {
    type Output = Vector<T>;

    fn add(self, rhs: Vector<T>) -> Self::Output {
        Vector::from_raw(self.raw.borrow().add_vec(rhs.raw.borrow().as_ref()))
    }
}

impl <T> Mul<T> for Vector<T> {
    type Output = Vector<T>;

    fn mul(self, rhs: T) -> Self::Output {
        //TODO lazy / deferred
        Vector::from_raw(self.raw.borrow().mult_with_scalar(rhs))
    }
}

impl <T> SubAssign<Vector<T>> for Vector<T> {
    fn sub_assign(&mut self, rhs: Vector<T>) {
        assert_eq!(self.dim(), rhs.dim());
        self.raw.borrow_mut().minus_vec_in_place(rhs.raw.borrow().as_ref())
    }
}




trait LinAlgMatrix<T> {
    fn as_any(&self) -> &dyn Any;

    fn num_rows(&self) -> usize;
    fn num_cols(&self) -> usize;

    fn box_clone(&self) -> Box<dyn LinAlgMatrix<T>>;

    fn get(&self, row: usize, col: usize) -> T;
    fn set(&mut self, row: usize, col: usize, new_value: T);

    fn mult_with_vector(&self, rhs: &dyn LinAlgVector<T>) -> Box<dyn LinAlgVector<T>>;
    fn mult_with_scalar(&self, rhs: T) -> Box<dyn LinAlgMatrix<T>>;

    fn plus_matrix(&self, rhs: &dyn LinAlgMatrix<T>) -> Box<dyn LinAlgMatrix<T>>;
    fn minus_matrix(&self, rhs: &dyn LinAlgMatrix<T>) -> Box<dyn LinAlgMatrix<T>>;
    fn plus_matrix_in_place(&mut self, rhs: &dyn LinAlgMatrix<T>);
    fn minus_matrix_in_place(&mut self, rhs: &dyn LinAlgMatrix<T>);

    fn transposed_times_vector(&self, rhs: &dyn LinAlgVector<T>) -> Box<dyn LinAlgVector<T>>; //TODO 'program' built from 'primitives'
}

trait LinAlgVector<T>: Debug {
    fn as_any(&self) -> &dyn Any;

    fn dim(&self) -> usize;

    fn box_clone(&self) -> Box<dyn LinAlgVector<T>>;

    fn get(&self, i: usize) -> T;
    fn set(&mut self, i: usize, new_value: T);

    fn multiply_element_wise(&self, rhs: &dyn LinAlgVector<T>) -> Box<dyn LinAlgVector<T>>;
    fn map(&self, f: &dyn Fn(T) -> T) -> Box<dyn LinAlgVector<T>>;
    fn mult_with_scalar(&self, scalar: T) -> Box<dyn LinAlgVector<T>>;

    fn add_vec(&self, rhs: &dyn LinAlgVector<T>) -> Box<dyn LinAlgVector<T>>;
    fn mult_with_transposed_vec(&self, rhs: &dyn LinAlgVector<T>) -> Box<dyn LinAlgMatrix<T>>;

    fn minus_vec_in_place(&mut self, rhs: &dyn LinAlgVector<T>);
}




