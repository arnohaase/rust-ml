use std::fmt::{Debug, Formatter};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::RwLock;

use lazy_static::lazy_static;
use triomphe::Arc;


#[derive(Clone, Eq, PartialEq, Debug, Copy)]
pub enum DimensionKind {
    Regular,
    Collection,
    Gradient,
    Polynomial,
    //TODO RGB etc.
}

#[derive(Clone, Eq, PartialEq, Debug, Copy)]
pub struct Dimension {
    pub len: usize,
    pub kind: DimensionKind,
}


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
    dimensions: Vec<Dimension>,
    buf: Arc<RwLock<Vec<f64>>>,
}
impl Debug for Tensor {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self.dimensions().len() {
            0 => write!(f, "{}", self.buf.read().unwrap()[0]),
            1 => write!(f, "{:?}", self.buf.read().unwrap()),
            _ => write_rec(f, &self.buf().read().unwrap(), self.dimensions()),
        }
    }
}

fn write_rec(f: &mut Formatter<'_>, buf: &[f64], dimensions: &[Dimension]) -> std::fmt::Result {
    if dimensions.len() == 1 {
        write!(f, "{:?}", buf)
    }
    else {
        write!(f, "[")?;
        let inner_dims = &dimensions[1..];
        let chunk_size: usize = inner_dims.iter().map(|d| d.len).product();
        for chunk in buf.chunks(chunk_size) {
            write_rec(f, chunk, inner_dims)?;
        }
        write!(f, "]")
    }
}


impl Tensor {
    pub fn from_raw(dimensions: Vec<Dimension>, buf: Vec<f64>) -> Tensor {
        Tensor {
            id: new_tensor_id(),
            version: Default::default(),
            dimensions,
            buf: Arc::new(RwLock::new(buf)),
        }
    }

    pub fn scalar(x: f64) -> Tensor {
        Self::from_raw(vec![], vec![x])
    }

    pub fn vector(xs: Vec<f64>, kind: DimensionKind) -> Tensor {
        Self::from_raw(vec![Dimension {
            len: xs.len(),
            kind,
        }], xs)
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
    pub fn is_vector(&self) -> bool {
        self.dimensions.len() == 1
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
    pub fn dimensions(&self) -> &[Dimension] {
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

    /// This is largely for testing: It checks if two tensors have the same geometry and 'pretty
    ///  much' the same elements, i.e. the same elements within typical rounding errors. The margin
    ///  for rounding errors is pretty lax - this is meant for verifying program logic, not
    ///  numerical accuracy
    #[must_use]
    pub fn is_pretty_much_equal_to(&self, other: &Tensor) -> bool {
        const THRESHOLD: f64 = 1e-5;

        if self.dimensions() != other.dimensions() {
            return false;
        }
        let buf_a = self.buf().read().unwrap();
        let buf_b = other.buf().read().unwrap();
        for i in 0..buf_a.len() {
            if (buf_a[i] - buf_b[i]).abs() > THRESHOLD {
                return false;
            }
        }
        true
    }

    pub fn assert_pretty_much_equal_to(&self, other: &Tensor) {
        if !self.is_pretty_much_equal_to(other) {
            panic!("{:?} != {:?}", self, other);
        }
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

