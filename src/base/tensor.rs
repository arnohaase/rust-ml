use std::fmt::Debug;
use crate::base::element_wise::ElementWise;

pub trait Tensor: Clone + Debug + ElementWise {
    fn id(&self) -> u32;
    fn version(&self) -> u32;
}
