


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

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Dimensions {
    dimensions: Vec<Dimension>,
}
impl Dimensions {
    pub fn new(dimensions: Vec<Dimension>) -> Dimensions {
        Dimensions { dimensions }
    }

    pub fn raw(&self) -> &[Dimension] {
        &self.dimensions
    }

    /// The number of elements spanned by this dimensions, i.e. the product of all dimension sizes
    pub fn size(&self) -> usize {
        self.size_without_outer(0)
    }

    /// the 'size' of the n outermost dimensions, or - equivalently - the number of chunks
    ///  consisting of the (len-n) inner dimensions
    pub fn size_outer(&self, num_outer_dims: usize) -> usize {
        self.dimensions[..num_outer_dims].iter()
            .map(|d| d.len)
            .product()
    }

    /// The size of an 'inner' tensor, i.e. without n outer dimensions
    pub fn size_without_outer(&self, num_outer_dims: usize) -> usize {
        self.dimensions[num_outer_dims..].iter()
            .map(|d| d.len)
            .product()
    }

    /// The size of an inner tensor consisting of the innermost n dimensions
    pub fn size_inner(&self, num_inner_dims: usize) -> usize {
        self.size_without_outer(self.dimensions.len() - num_inner_dims)
    }
}

impl From<Vec<Dimension>> for Dimensions {
    fn from(value: Vec<Dimension>) -> Self {
        Dimensions::new(value)
    }
}


#[cfg(test)]
mod test {
    use rstest::rstest;
    use crate::dimension::{Dimension, DimensionKind, Dimensions};

    #[rstest]
    #[case(vec![], 1)]
    #[case(vec![1], 1)]
    #[case(vec![2], 2)]
    #[case(vec![1, 1], 1)]
    #[case(vec![2, 3], 6)]
    #[case(vec![1, 2, 3, 4, 5, 6], 720)]
    fn test_size(#[case] dimensions: Vec<usize>, #[case] expected_size: usize) {
        let len = dimensions.len();
        let dimensions: Dimensions = dimensions.iter()
            .map(|&len| Dimension { len, kind: DimensionKind::Regular })
            .collect::<Vec<_>>()
            .into();

        assert_eq!(dimensions.size(), expected_size);
        assert_eq!(dimensions.size_outer(len), expected_size);
        assert_eq!(dimensions.size_without_outer(0), expected_size);
        assert_eq!(dimensions.size_inner(len), expected_size);
    }

    #[rstest]
    fn test_size_outer() {
        todo!()
    }

    #[rstest]
    fn test_size_without_outer() {
        todo!()
    }

    #[rstest]
    fn test_size_inner() {
        todo!()
    }
}