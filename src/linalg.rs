use std::ops::{Add, DerefMut, Index, IndexMut, Mul, SubAssign};
use rand::random;

#[derive(Clone, Debug)]
pub struct Matrix {
    pub col_dim: usize,
    pub row_dim: usize,
    pub coefficients: Vec<f64>,
}
impl Matrix {
    pub fn new_zero(num_rows: usize, num_cols: usize) -> Matrix {
        let mut coefficients = Vec::with_capacity(num_rows * num_cols);
        for _ in 0..num_rows * num_cols {
            coefficients.push(0.0);
        }

        Matrix {
            col_dim: num_cols,
            row_dim: num_rows,
            coefficients
        }
    }

    pub fn new_random(num_rows: usize, num_cols: usize) -> Matrix {
        let mut coefficients = Vec::with_capacity(num_rows * num_cols);
        for _ in 0..num_rows * num_cols {
            coefficients.push(random::<f64>() * 2.0 - 1.0);
        }

        Matrix {
            col_dim: num_cols,
            row_dim: num_rows,
            coefficients
        }
    }

    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.coefficients[row * self.col_dim + col]
    }


    pub fn transposed_times(&self, rhs: &Vector) -> Vector {
        assert_eq!(self.row_dim, rhs.dim());

        let mut result = Vector::new_zero(self.col_dim);

        for row in 0..self.row_dim {
            for col in 0..self.col_dim {
                result[col] += self[row][col] * rhs[row];
            }
        }

        result
    }
}

impl SubAssign<&Matrix> for Matrix {
    fn sub_assign(&mut self, rhs: &Matrix) {
        assert_eq!(self.row_dim, rhs.row_dim);
        assert_eq!(self.col_dim, rhs.col_dim);

        for i in 0..self.coefficients.len() {
            self.coefficients[i] -= rhs.coefficients[i];
        }
    }
}

impl Index<usize> for Matrix {
    type Output = [f64];

    fn index(&self, row: usize) -> &Self::Output {
        &self.coefficients[row * self.col_dim.. (row+1) * self.col_dim]
    }
}

impl Mul<&Vector> for &Matrix {
    type Output = Vector;

    fn mul(self, rhs: &Vector) -> Self::Output {
        assert_eq!(self.col_dim, rhs.dim());

        let mut result = Vector::new_zero(self.row_dim);

        for row in 0..self.row_dim {
            for col in 0..self.col_dim {
                result[row] += self[row][col] * rhs[col];
            }
        }

        result
    }
}

impl Mul<f64> for &Matrix {
    type Output = Matrix;

    fn mul(self, rhs: f64) -> Self::Output {
        let mut result = self.clone();
        for i in 0..result.coefficients.len() {
            result.coefficients[i] *= rhs;
        }
        result
    }
}



#[derive(Clone, Debug)]
pub struct Vector {
    pub values: Vec<f64>,
}
impl Vector {
    pub fn new_zero(dim: usize) -> Vector {
        let mut values = Vec::with_capacity(dim);
        for _ in 0..dim {
            values.push(0.0);
        }

        assert_eq!(dim, values.len());
        Vector {
            values
        }
    }

    pub fn new_random(dim: usize) -> Vector {
        let mut values = Vec::with_capacity(dim);
        for _ in 0..dim {
            values.push(random::<f64>() * 2.0 - 1.0);
        }

        assert_eq!(dim, values.len());
        Vector {
            values
        }
    }

    pub fn dim(&self) -> usize {
        return self.values.len()
    }

    pub fn multiply_element_wise(&self, rhs: &Vector) -> Vector {
        assert_eq!(self.dim(), rhs.dim());

        let mut result = self.clone();
        for i in 0..result.dim() {
            result.values[i] *= rhs.values[i];
        }
        result
    }

    pub fn map(&self, f: impl Fn(f64) -> f64) -> Vector {
        let mut result = self.clone();
        for i in 0..result.dim() {
            result[i] = f(result[i]);
        }
        result
    }
}

impl Mul<f64> for &Vector {
    type Output = Vector;

    fn mul(self, rhs: f64) -> Self::Output {
        let mut result = self.clone();
        for i in 0..result.values.len() {
            result.values[i] *= rhs;
        }
        result
    }
}

impl SubAssign<&Vector> for Vector {
    fn sub_assign(&mut self, rhs: &Vector) {
        assert_eq!(self.dim(), rhs.dim());

        for i in 0..self.dim() {
            self.values[i] -= rhs.values[i];
        }
    }
}

impl Add<&Vector> for Vector {
    type Output = Vector;

    fn add(self, rhs: &Vector) -> Self::Output {
        assert_eq!(self.dim(), rhs.dim());
        let mut result = self.clone();
        for i in 0..rhs.dim() {
            result[i] += rhs[i];
        }
        result
    }
}

impl Index<usize> for Vector {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        &self.values[index]
    }
}

impl IndexMut<usize> for Vector {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let buf = self.values.deref_mut();
        &mut buf[index]
    }
}
