use std::fmt::Debug;

pub trait Tensor: Clone + Debug {
    fn id(&self) -> u32;
    fn version(&self) -> u32;
}
