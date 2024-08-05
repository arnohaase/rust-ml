use std::fmt::{Debug, Formatter};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::RwLock;

use lazy_static::lazy_static;
use triomphe::Arc;


#[derive(Clone, Eq, PartialEq, Debug, Copy)]
pub enum DimensionKind {
    Regular,
    Collection,
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
        Tensor {
            env: &self,
            id: new_tensor_id(),
            version: Default::default(),
            dimensions,
            buf: Arc::new(RwLock::new(buf)),
        }
    }
}



pub struct Tensor<'env, E: TensorEnv + ?Sized> {
    env: &'env E,
    id: u32,
    version: Arc<AtomicU32>,
    //TODO do we need stride information for autograd stuff? if so, where to put it?
    dimensions: Vec<Dimension>,
    buf: E::Buffer,
}
impl <'env, E: TensorEnv> Clone for Tensor<'env, E> {
    fn clone(&self) -> Self {
        Tensor {
            env: self.env,
            id: self.id,
            version: self.version.clone(),
            dimensions: self.dimensions.clone(),
            buf: self.buf.clone(),
        }
    }
}

impl <'env> Debug for Tensor<'env, BlasEnv> {
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


impl <'env, E: TensorEnv> Tensor<'env, E> {

    pub fn is_scalar(&self) -> bool {
        self.dimensions.is_empty()
    }
    pub fn is_vector(&self) -> bool {
        self.dimensions.len() == 1
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

    pub fn env(&self) -> &'env E {
        self.env
    }

    pub fn clone_with_new_id(&self) -> Tensor<'env, E> {
        let mut result = self.clone();
        result.id = new_tensor_id();
        result
    }
}

impl <'env> Tensor<'env, BlasEnv> {
    pub fn buf(&self) -> &RwLock<Vec<f64>> {
        self.buf.as_ref()
    }

    /// This is largely for testing: It checks if two tensors have the same geometry and 'pretty
    ///  much' the same elements, i.e. the same elements within typical rounding errors. The margin
    ///  for rounding errors is pretty lax - this is meant for verifying program logic, not
    ///  numerical accuracy
    #[must_use]
    pub fn is_pretty_much_equal_to(&self, other: &Tensor<'env, BlasEnv>) -> bool {
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

    pub fn assert_pretty_much_equal_to(&self, other: &Tensor<'env, BlasEnv>) {
        if !self.is_pretty_much_equal_to(other) {
            panic!("{:?} != {:?}", self, other);
        }
    }
}
