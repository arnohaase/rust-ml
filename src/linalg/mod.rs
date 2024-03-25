pub mod naive;

use std::any::Any;
use std::fmt::{Debug, Display};
use std::ops::{Add, Div, Mul, Sub, SubAssign};



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
    raw: Box<dyn RawMatrix<T>>,
}

impl <T> Matrix<T> {
    pub fn from_raw(raw: Box<dyn RawMatrix<T>>) -> Matrix<T> {
        Matrix {
            raw
        }
    }

    pub fn num_rows(&self) -> usize {
        self.raw.num_rows()
    }
    pub fn num_cols(&self) -> usize {
        self.raw.num_cols()
    }

    pub fn get(&self, row: usize, col: usize) -> T {
        self.raw.get(row, col)
    }
    pub fn set(&mut self, row: usize, col: usize, new_value: T) {
        self.raw.set(row, col, new_value)
    }

    //TODO 'program' built from 'primitives'
    pub fn transposed_times_vector(&self, rhs: &Vector<T>) -> Vector<T> {
        Vector::from_raw(self.raw.transposed_times_vector(rhs.raw.as_ref()))
    }
}


impl <T> Mul<&Vector<T>> for &Matrix<T> {
    type Output = Vector<T>;

    fn mul(self, rhs: &Vector<T>) -> Self::Output {
        assert_eq!(self.num_cols(), rhs.dim());
        //TODO lazy / deferred
        Vector::from_raw(self.raw.mult_with_vector(rhs.raw.as_ref()))
    }
}

//TODO in-place for ownership based variant

impl <T> Mul<T> for &Matrix<T> {
    type Output = Matrix<T>;

    fn mul(self, rhs: T) -> Self::Output {
        //TODO lazy / deferred
        Matrix::from_raw(self.raw.mult_with_scalar(rhs))
    }
}

impl <T> SubAssign<&Matrix<T>> for Matrix<T> {
    fn sub_assign(&mut self, rhs: &Matrix<T>) {
        assert_eq!(self.num_rows(), rhs.num_rows());
        assert_eq!(self.num_cols(), rhs.num_cols());

        self.raw.minus_matrix_in_place(rhs.raw.as_ref())
    }
}


#[derive(Debug)]
pub struct Vector<T> {
    raw: Box<dyn RawVector<T>>,
}
impl <T> Vector<T> {
    fn from_raw(raw: Box<dyn RawVector<T>>) -> Vector<T> {
        Vector {
            raw
        }
    }

    pub fn get(&self, index: usize) -> T {
        self.raw.get(index)
    }
    pub fn set(&mut self, index: usize, new_value: T) {
        self.raw.set(index, new_value)
    }

    pub fn dim(&self) -> usize {
        self.raw.dim()
    }

    //TODO pass references for non-consumed parameters?
    pub fn mult_with_transposed_vec(&self, rhs: &Vector<T>) -> Matrix<T> {
        Matrix::from_raw(self.raw.mult_with_transposed_vec(rhs.raw.as_ref()))
    }

    pub fn map(&self, f: impl Fn(T) -> T) -> Vector<T> {
        Vector::from_raw(self.raw.map(&f))
    }

    pub fn multiply_element_wise(&self, rhs: &Vector<T>) -> Vector<T> {
        Vector::from_raw(self.raw.multiply_element_wise(rhs.raw.as_ref()))
    }
}

impl <T> Clone for Vector<T> {
    fn clone(&self) -> Self {
        Vector::from_raw(self.raw.box_clone())
    }
}

//TODO syntactic sugar for indexed get / set syntax

impl <T> Add<&Vector<T>> for &Vector<T> {
    type Output = Vector<T>;

    fn add(self, rhs: &Vector<T>) -> Self::Output {
        Vector::from_raw(self.raw.add_vec(rhs.raw.as_ref()))
    }
}

impl <T> Mul<T> for &Vector<T> {
    type Output = Vector<T>;

    fn mul(self, rhs: T) -> Self::Output {
        //TODO lazy / deferred
        Vector::from_raw(self.raw.mult_with_scalar(rhs))
    }
}

impl <T> SubAssign<&Vector<T>> for Vector<T> {
    fn sub_assign(&mut self, rhs: &Vector<T>) {
        assert_eq!(self.dim(), rhs.dim());
        self.raw.minus_vec_in_place(rhs.raw.as_ref())
    }
}




trait RawMatrix<T> {
    fn as_any(&self) -> &dyn Any;

    fn num_rows(&self) -> usize;
    fn num_cols(&self) -> usize;

    fn box_clone(&self) -> Box<dyn RawMatrix<T>>;

    fn get(&self, row: usize, col: usize) -> T;
    fn set(&mut self, row: usize, col: usize, new_value: T);

    fn mult_with_vector(&self, rhs: &dyn RawVector<T>) -> Box<dyn RawVector<T>>;
    fn mult_with_scalar(&self, rhs: T) -> Box<dyn RawMatrix<T>>;

    fn plus_matrix(&self, rhs: &dyn RawMatrix<T>) -> Box<dyn RawMatrix<T>>;
    fn minus_matrix(&self, rhs: &dyn RawMatrix<T>) -> Box<dyn RawMatrix<T>>;
    fn plus_matrix_in_place(&mut self, rhs: &dyn RawMatrix<T>);
    fn minus_matrix_in_place(&mut self, rhs: &dyn RawMatrix<T>);

    fn transposed_times_vector(&self, rhs: &dyn RawVector<T>) -> Box<dyn RawVector<T>>; //TODO 'program' built from 'primitives'
}

trait RawVector<T>: Debug {
    fn as_any(&self) -> &dyn Any;

    fn dim(&self) -> usize;

    fn box_clone(&self) -> Box<dyn RawVector<T>>;

    fn get(&self, i: usize) -> T;
    fn set(&mut self, i: usize, new_value: T);

    fn multiply_element_wise(&self, rhs: &dyn RawVector<T>) -> Box<dyn RawVector<T>>;
    fn map(&self, f: &dyn Fn(T) -> T) -> Box<dyn RawVector<T>>;
    fn mult_with_scalar(&self, scalar: T) -> Box<dyn RawVector<T>>;

    fn add_vec(&self, rhs: &dyn RawVector<T>) -> Box<dyn RawVector<T>>;
    fn mult_with_transposed_vec(&self, rhs: &dyn RawVector<T>) -> Box<dyn RawMatrix<T>>;

    fn minus_vec_in_place(&mut self, rhs: &dyn RawVector<T>);
}




