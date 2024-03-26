use std::fmt::{Debug, Display, Formatter};
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Sub};
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use triomphe::Arc;

pub trait Float:
    Copy + Display + Debug +
    Add<Output=Self> + Sub<Output=Self> + Mul<Output=Self> + Div<Output=Self>
{}

impl Float for f64 {}

pub struct TensorEnv<F: Float> {
    pd: PhantomData<F>,
}
impl <F: Float> TensorEnv<F> {
    pub fn new() -> TensorEnv<F> {
        TensorEnv {
            pd: Default::default(),
        }
    }

    fn assemble(&self, geometry: Geometry, data: Vec<F>) -> Tensor<F> {
        Tensor {
            env: self,
            geometry,
            data: Arc::new(RwLock::new(data)),
        }
    }

    pub fn create_scalar(&self, value: F) -> Tensor<F> {
        self.assemble(Geometry::Scalar,vec![value])
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Geometry { //TODO
    Scalar,
    Vector(usize),
    // Matrix,
}

#[derive(Clone)]
pub struct Tensor<'env, F: Float> {
    env: &'env TensorEnv<F>,
    geometry: Geometry,
    data: Arc<RwLock<Vec<F>>>,
}
impl<'env, F: Float> Debug for Tensor<'env, F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}: {:?}", self.geometry, self.read())
    }
}


impl<'env, F: Float> Tensor<'env, F> {
    pub fn r(&self) -> Tensor<'env, F> {
        self.clone()
    }

    fn read(&self) -> RwLockReadGuard<'_, Vec<F>> {
        self.data.read().unwrap()
    }
    fn write(&self) -> RwLockWriteGuard<'_, Vec<F>> {
        self.data.write().unwrap()
    }


    pub fn plus(&self, other: Tensor<'_, F>) -> Tensor<'env, F> {
        assert_eq!(self.geometry, other.geometry);
        let v1 = self.read();
        let v2 = other.read();

        let mut data = Vec::with_capacity(v1.len());
        for i in 0..v1.len() {
            data.push(v1[i] + v2[i]);
        }
        self.env.assemble(self.geometry, data)
    }

    pub fn scalar_mult(&self, scalar: Tensor<'_, F>) -> Tensor<'env, F> { //TODO generalize - this is just to get an initial example to work
        assert_eq!(scalar.geometry, Geometry::Scalar);

        let scalar = scalar.read()[0];
        let v1 = self.read();

        let mut data = Vec::with_capacity(v1.len());
        for &x in v1.iter() {
            data.push(x*scalar);
        }
        self.env.assemble(self.geometry, data)
    }

    pub fn pow_int(&self, exp: usize) -> Tensor<'env, F> {
        let v1 = self.read();
        let mut data = Vec::with_capacity(v1.len());
        for &x in v1.iter() {
            let mut value = x;
            for _ in 1..exp { //TODO this is hacky and inefficient, to get an example running early on
                value = value * x;
            }
            data.push(value);
        }
        self.env.assemble(self.geometry, data)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_play_around() {
        let env = TensorEnv::new();

        let a = env.create_scalar(2.0);
        let b = env.create_scalar(3.0);

        let f = a.scalar_mult(b);

        println!("{:?}", f);

    }
}






















