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
pub enum Repr {
    ZERO,
    ONE,
    BLAS { //TODO generalize to wgpu etc.
        buf: Arc<RwLock<Vec<f64>>>,
    }
}

#[derive(Clone)]
pub struct Tensor {
    id: u32,
    version: Arc<AtomicU32>,
    //TODO do we need stride information for autograd stuff? if so, where to put it?
    dimensions: Vec<usize>,
    repr: Repr,
}
impl Tensor {
    pub fn from_raw(dimensions: Vec<usize>, buf: Vec<f64>) -> Tensor {
        Tensor {
            id: new_tensor_id(),
            version: Default::default(),
            dimensions,
            repr: Repr::BLAS { buf: Arc::new(RwLock::new(buf)) },
        }
    }

    pub fn zero(dimensions: Vec<usize>) -> Tensor {
        Tensor {
            id: new_tensor_id(),
            version: Default::default(),
            dimensions,
            repr: Repr::ZERO,
        }
    }

    pub fn one(first_half_dimensions: &[usize]) -> Tensor {
        let mut dimensions: Vec<usize> = first_half_dimensions.into();
        for i in (0..dimensions.len()).rev() {
            dimensions.push(dimensions[i]);
        }
        Tensor {
            id: new_tensor_id(),
            version: Default::default(),
            dimensions,
            repr: Repr::ONE,
        }
    }

    pub fn is_zero(&self) -> bool {
        matches!(self.repr, Repr::ZERO)
    }

    pub fn is_one(&self) -> bool {
        matches!(self.repr, Repr::ONE)
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

    pub fn repr(&self) -> &Repr {
        &self.repr
    }
}

#[cfg(test)]
mod test {
    use rstest::rstest;
    use crate::n::tensor::Tensor;

    #[rstest]
    #[case(&[], &[])]
    #[case(&[1], &[1, 1])]
    #[case(&[2], &[2, 3])]
    #[case(&[4, 5], &[4, 5, 5, 4])]
    fn test_one(#[case] first_half_dim: &[usize], #[case] expected_dim: &[usize]) {
        let one = Tensor::one(first_half_dim);
        assert!(one.is_one());
        assert_eq!(expected_dim, one.dimensions);
    }
}

