use std::sync::RwLock;
use triomphe::Arc;
use crate::tensor::{Dimension, DimensionKind, new_tensor_id, Tensor};

pub trait TensorEnv {
    type Buffer: Clone;

    fn create_tensor(&self, dimensions: Vec<Dimension>, buf: Vec<f64>) -> Tensor<Self>;

    fn scalar(&self, x: f64) -> Tensor<Self> {
        self.create_tensor(vec![], vec![x])
    }

    fn vector(&self, xs: Vec<f64>, kind: DimensionKind) -> Tensor<Self> {
        self.create_tensor(vec![Dimension {
            len: xs.len(),
            kind,
        }], xs)
    }
}


pub struct BlasEnv {
}

impl TensorEnv for BlasEnv {
    type Buffer = Arc<RwLock<Vec<f64>>>;

    fn create_tensor(&self, dimensions: Vec<Dimension>, buf: Vec<f64>) -> Tensor<Self> {
        Tensor::create_from_raw(&self, dimensions, Arc::new(RwLock::new(buf)))
    }
}


pub struct WgpuEnv {

}

impl WgpuEnv {
    pub fn new() -> WgpuEnv {
        todo!()
    }
}

impl TensorEnv for WgpuEnv {
    type Buffer = ();

    fn create_tensor(&self, dimensions: Vec<Dimension>, buf: Vec<f64>) -> Tensor<Self> {
        todo!()
    }
}








