use std::fmt::Debug;

pub trait Tensor: Clone + Debug {
    fn id(&self) -> u32;
    fn version(&self) -> u32;
    fn with_data<X>(&self, f: impl FnOnce(&[f64]) -> X) -> X;
}