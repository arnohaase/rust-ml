use std::cmp::Ordering;

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

    pub fn num_dims(&self) -> usize {
        self.dimensions.len()
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

    /// Compares two tensors' dimensions, checking if one of the tensors contains parts with the
    ///  other's dimensions
    pub fn match_with_other(&self, other: &Dimensions) -> MatchDimensionsResult {
        //TODO ambiguity?
        //TODO for nested_dims in 0..lhs.len() - rhs.len()

        match self.dimensions.len().cmp(&other.dimensions.len()) {
            Ordering::Equal =>
                if self == other { MatchDimensionsResult::Equal } else { MatchDimensionsResult::Mismatch }
            Ordering::Less =>
                Self::check_dims_contained(other.raw(), self.raw(), |num_wrapper_dims, num_nested_dims| MatchDimensionsResult::RightContainsLeft { num_wrapper_dims, num_nested_dims, }),
            Ordering::Greater =>
                Self::check_dims_contained(self.raw(), other.raw(), |num_wrapper_dims, num_nested_dims| MatchDimensionsResult::LeftContainsRight { num_wrapper_dims, num_nested_dims, }),
        }
    }

    fn check_dims_contained(
        longer: &[Dimension],
        shorter: &[Dimension],
        factory: impl FnOnce(usize, usize) -> MatchDimensionsResult,
    ) -> MatchDimensionsResult {
        for num_nested_dims in 0..= longer.len() - shorter.len() {
            if longer[..longer.len()- num_nested_dims].ends_with(shorter) {
                return factory(longer.len() - shorter.len() - num_nested_dims, num_nested_dims);
            }
        }
        MatchDimensionsResult::Mismatch
    }
}

impl From<Vec<Dimension>> for Dimensions {
    fn from(value: Vec<Dimension>) -> Self {
        Dimensions::new(value)
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum MatchDimensionsResult {
    Equal,
    Mismatch,
    LeftContainsRight { num_wrapper_dims: usize, num_nested_dims: usize, },
    RightContainsLeft { num_wrapper_dims: usize, num_nested_dims: usize, },
}


#[cfg(test)]
mod test {
    use rstest::rstest;
    use crate::dimension::{Dimension, DimensionKind, Dimensions, MatchDimensionsResult};

    #[rstest]
    #[case(vec![], 1)]
    #[case(vec![1], 1)]
    #[case(vec![2], 2)]
    #[case(vec![1, 1], 1)]
    #[case(vec![2, 3], 6)]
    #[case(vec![1, 2, 3, 4, 5, 6], 720)]
    fn test_size(#[case] dimensions: Vec<usize>, #[case] expected_size: usize) {
        let len = dimensions.len();
        let dimensions = dims_from_sizes(dimensions);

        assert_eq!(dimensions.size(), expected_size);
        assert_eq!(dimensions.size_outer(len), expected_size);
        assert_eq!(dimensions.size_without_outer(0), expected_size);
        assert_eq!(dimensions.size_inner(len), expected_size);
    }

    fn dims_from_sizes(sizes: Vec<usize>) -> Dimensions {
        sizes.iter()
            .map(|&len| Dimension { len, kind: DimensionKind::Regular })
            .collect::<Vec<_>>()
            .into()
    }

    #[rstest]
    #[case(vec![], 0, 1)]
    #[case(vec![3], 0, 1)]
    #[case(vec![3], 1, 3)]
    #[case(vec![3, 4], 0, 1)]
    #[case(vec![3, 4], 1, 3)]
    #[case(vec![3, 4], 2, 12)]
    fn test_size_outer(#[case] dimensions: Vec<usize>, #[case] n: usize, #[case] expected: usize) {
        let dimensions = dims_from_sizes(dimensions);
        assert_eq!(dimensions.size_outer(n), expected);
    }

    #[rstest]
    #[case(vec![], 0, 1)]
    #[case(vec![3], 0, 3)]
    #[case(vec![3], 1, 1)]
    #[case(vec![3, 4], 0, 12)]
    #[case(vec![3, 4], 1, 4)]
    #[case(vec![3, 4], 2, 1)]
    fn test_size_without_outer(#[case] dimensions: Vec<usize>, #[case] n: usize, #[case] expected: usize) {
        let dimensions = dims_from_sizes(dimensions);
        assert_eq!(dimensions.size_without_outer(n), expected);
    }

    #[rstest]
    #[case(vec![], 0, 1)]
    #[case(vec![3], 0, 1)]
    #[case(vec![3], 1, 3)]
    #[case(vec![3, 4], 0, 1)]
    #[case(vec![3, 4], 1, 4)]
    #[case(vec![3, 4], 2, 12)]
    fn test_size_inner(#[case] dimensions: Vec<usize>, #[case] n: usize, #[case] expected: usize) {
        let dimensions = dims_from_sizes(dimensions);
        assert_eq!(dimensions.size_inner(n), expected);
    }


    fn dims(spec: &str) -> Dimensions {
        if spec.is_empty() {
            return vec![].into();
        }

        let dims_spec = spec.split('-').collect::<Vec<_>>();
        dims_spec.iter()
            .map(|s| {
                let kind = match s.chars().next().unwrap() {
                    'R' => DimensionKind::Regular,
                    'C' => DimensionKind::Collection,
                    'P' => DimensionKind::Polynomial,
                    _ => unimplemented!(),
                };
                let len: usize = s[1..].parse().unwrap();
                Dimension { len, kind, }
            })
            .collect::<Vec<_>>()
            .into()
    }

    #[rstest]
    #[case::scalar_scalar("", "", MatchDimensionsResult::Equal)]
    #[case::scalar_vec("", "C3", MatchDimensionsResult::RightContainsLeft {num_wrapper_dims: 1, num_nested_dims: 0})]
    #[case::vec_scalar("C3", "", MatchDimensionsResult::LeftContainsRight {num_wrapper_dims: 1, num_nested_dims: 0})]
    fn test_match_with_other(#[case] dim_a_spec: &str, #[case] dim_b_spec: &str, #[case] expected: MatchDimensionsResult) {
        assert_eq!(dims(dim_a_spec).match_with_other(&dims(dim_b_spec)), expected);
    }

    #[test]
    fn test_fit_dimensions() {
        todo!()
    }

}