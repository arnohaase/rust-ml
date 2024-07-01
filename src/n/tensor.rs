use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::RwLock;

use lazy_static::lazy_static;
use triomphe::Arc;

lazy_static! {
    static ref TENSOR_ID_COUNTER: AtomicU32 = AtomicU32::new(0);
}
pub fn new_tensor_id() -> u32 {
    TENSOR_ID_COUNTER.fetch_add(1, Ordering::SeqCst)
}

#[derive(Clone)]
pub struct Tensor {
    id: u32,
    version: Arc<AtomicU32>,
    //TODO do we need stride information for autograd stuff? if so, where to put it?
    dimensions: Vec<usize>,
    buf: Arc<RwLock<Vec<f64>>>,
}
impl Tensor {
    pub fn from_raw(dimensions: Vec<usize>, buf: Vec<f64>) -> Tensor {
        Tensor {
            id: new_tensor_id(),
            version: Default::default(),
            dimensions,
            buf: Arc::new(RwLock::new(buf)),
        }
    }

    pub fn zero() -> Tensor {
        Self::from_raw(vec![], vec![0.0])
    }

    pub fn one() -> Tensor {
        Self::from_raw(vec![], vec![1.0])
    }

    pub fn is_scalar(&self) -> bool {
        self.dimensions.is_empty()
    }

    pub fn is_zero(&self) -> bool {
        self.is_scalar() && self.buf.read().unwrap()[0] == 0.0
    }

    pub fn is_one(&self) -> bool {
        self.is_scalar() && self.buf.read().unwrap()[0] == 1.0
    }

    pub fn id(&self) -> u32 {
        self.id
    }
    pub fn version(&self) -> u32 {
        self.version.fetch_add(0, Ordering::Acquire)
    }
    pub fn dimensions(&self) -> &[usize] {
        &self.dimensions
    }

    pub fn buf(&self) -> &RwLock<Vec<f64>> {
        self.buf.as_ref()
    }

    pub fn clone_with_new_id(&self) -> Tensor {
        let mut result = self.clone();
        result.id = new_tensor_id();
        result
    }
}

#[cfg(test)]
mod test {
    use crate::n::tensor::Tensor;

    #[test]
    fn test_one() {
        assert!(Tensor::one().is_one());
    }
}

