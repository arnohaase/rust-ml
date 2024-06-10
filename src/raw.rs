use blas::{dgemm, dgemv};

pub struct Vector<const N: usize> {
    data: [f64;N]
}
impl <const N: usize> Vector<N> {
    pub fn zero() -> Vector<N> {
        Vector {
            data: [0.0; N],
        }
    }

    pub fn initialized(f: impl Fn(usize) -> f64) -> Vector<N> {
        let mut data = [0.0; N];
        for i in 0..N {
            data[i] = f(i);
        }
        Vector { data }
    }

    /// self *= alpha
    ///
    /// This corresponds to the `dscal` BLAS function
    pub fn scale(&mut self, alpha: f64) {
        unsafe {
            blas::dscal(N as i32, alpha, &mut self.data, 1)
        }
    }

    /// self += alpha * x
    ///
    /// This corresponds to the `daxpy` BLAS function
    pub fn plus_alpha_x(&mut self, alpha: f64, x: &Vector<N>) {
        unsafe {
            blas::daxpy(N as i32, alpha, &x.data, 1, &mut self.data, 1);
        }
    }

    /// computes the scalar product of this Vector with another
    ///
    /// This corresponds to the `ddot` BLAS function
    pub fn dot(&self, y: &Vector<N>) -> f64 {
        unsafe {
            blas::ddot(N as i32, &self.data, 1, &y.data, 1)
        }
    }

    pub fn map(&mut self, f: impl Fn(f64) -> f64) {
        for i in 0..N {
            self.data[i] = f(self.data[i]);
        }
    }
}

pub struct DenseMatrix<const R: usize, const C: usize, const SIZE: usize> {
    data: [f64;SIZE],
}
impl <const R: usize, const C: usize, const SIZE: usize> DenseMatrix<R,C,SIZE> {
    pub fn zero() -> DenseMatrix<R,C,SIZE> {
        assert_eq!(SIZE, R*C);
        DenseMatrix {
            data: [0.0; SIZE],
        }
    }

    pub fn initialized(f: impl Fn(usize, usize) -> f64) -> DenseMatrix<R,C,SIZE> {
        assert_eq!(SIZE, R*C);
        let mut data = [0.0; SIZE];
        for row in 0..R {
            for col in 0..C {
                data[col*R + row] = f(row, col);
            }
        }
        DenseMatrix { data }
    }

    /// self += alpha * x
    ///
    /// add another matrix 'x' to self (with a scaling factor)
    pub fn plus_alpha_x(&mut self, alpha: f64, x: &DenseMatrix<R,C,SIZE>) {
        unsafe {
            blas::daxpy((R*C) as i32, alpha, &x.data, 1, &mut self.data, 1)
        }
    }

    /// y = alpha * self * x + beta * y
    ///
    /// This corresponds to the `dgemv` BLAS function
    pub fn alpha_a_x_plus_beta_y(&self, alpha: f64, x: &Vector<C>, beta: f64, y: &mut Vector<R>) {
        unsafe {
            dgemv(b'N', R as i32, C as i32, alpha, &self.data, C as i32, &x.data, 1, beta, &mut y.data, 1)
        }
    }

    /// C = alpha*A*B + beta*C
    ///
    /// This corresponds to the `dgemm` BLAS function
    pub fn alpha_a_b_plus_beta_c<const D: usize, const S1: usize, const S2: usize>(
        &mut self, alpha: f64, a: &DenseMatrix<R,D, S1>, b: &DenseMatrix<D,C, S2>, beta: f64
    ) {
        unsafe {
            dgemm(b'N', b'N', R as i32, C as i32, D as i32, alpha, &a.data, D as i32, &b.data, C as i32, beta, &mut self.data, C as i32)
        }
    }
}

pub struct Collection<T> {
    //TODO generalize to streams
    //TODO contraint T to 'Collectible'?
    //TODO scalar?
    values: Vec<T>
}
impl <T> Collection<T> {
    //TODO here or just as an operation?
    pub fn fold(&self, initial: f64, f: impl Fn(f64, &T) -> f64) -> f64 {
        self.values.iter()
            .fold(initial, f)
    }
}





