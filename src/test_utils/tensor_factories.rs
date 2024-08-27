use async_std::task::block_on;
use crate::tensor::{Dimension, DimensionKind, Tensor};
use crate::tensor_env::{BlasEnv, TensorEnv, WgpuEnv};


/// This function is meant for test fixtures: it parses a string representation for a tensor
///  and builds the tensor using the tensor environment that is passed in.
///
/// Tensor definitions strings look like this: `"R-R:[[1,2,3.5][6,7.1,8]]"` which would create a
///  two-dimensional tensor (Regular(2), Regular(3)).
///
/// The function does not bother with sophisticated error handling, it just panics if it cannot
///  parse a tensor specification.
pub fn tensor_from_spec<'a, E: TensorEnv>(spec: &str, env: &'a E) -> Tensor<'a, E> {
    if !spec.contains(":") {
        return env.scalar(spec.parse::<f64>().unwrap());
    }

    let s: Vec<&str> = spec.split(':').collect();
    assert_eq!(s.len(), 2);

    let kinds: Vec<DimensionKind> = s[0].split('-')
        .map(|k| match k {
            "R" => DimensionKind::Regular,
            "C" => DimensionKind::Collection,
            "P" => DimensionKind::Polynomial,
            _ => panic!("unsupported kind {k}"),
        })
        .collect();

    let mut offs = 0;
    let mut dimension_sizes = Vec::new();
    parse_dimension_sizes(s[1], kinds.len(), 0, &mut offs, &mut dimension_sizes);
    let dimension_sizes: Vec<usize> = dimension_sizes.iter()
        .map(|n| n.unwrap())
        .collect();

    let dimensions = kinds.iter()
        .zip(dimension_sizes.iter())
        .map(|(&kind, &len)| Dimension {
            len,
            kind,
        })
        .collect();

    let buf = parse_numbers(spec);

    env.create_tensor(dimensions, buf)
}

fn parse_numbers(spec: &str) -> Vec<f64> {
    let spec = &spec[spec.find(':').unwrap()+1 ..];

    let raw_nums: Vec<&str> = spec.split(&['[', ']', ',', ' '])
        .filter(|s| !s.is_empty())
        .collect();

    raw_nums.iter()
        .map(|s| s.parse::<f64>().unwrap())
        .collect()
}

fn parse_dimension_sizes(spec: &str, num_dims: usize, depth: usize, offs: &mut usize, result: &mut Vec<Option<usize>>) {
    //TODO scalars as a special case

    while result.len() < depth {
        result.push(None);
    }

    if depth == 0 {
        assert!(spec[*offs..].starts_with("["));
        *offs += 1;
        parse_dimension_sizes(spec, num_dims, depth+1, offs, result);
        assert!(spec[*offs..].starts_with("]"));
    }
    else if depth == num_dims {
        let numbers = spec[*offs..].split(']').next().unwrap();
        let num_numbers = numbers.split(',').count();
        match result[depth-1] {
            None => result[depth-1] = Some(num_numbers),
            Some(n) => assert_eq!(n, num_numbers),
        }
        *offs += numbers.len();
    }
    else {
        let mut count = 0;
        while spec[*offs..].starts_with("[") {
            *offs += 1;
            parse_dimension_sizes(spec, num_dims, depth+1, offs, result);
            assert!(spec[*offs..].starts_with("]"));
            *offs += 1;
            count += 1;
        }

        match result[depth-1] {
            None => result[depth-1] = Some(count),
            Some(n) => assert_eq!(n, count, "mismatch at depth {depth}"),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::tensor::{Dimension, DimensionKind};
    use crate::tensor_env::{BlasEnv, TensorEnv};
    use crate::test_utils::tensor_factories::tensor_from_spec;

    #[test]
    fn test_tensor_from_spec() {
        let env = BlasEnv{};

        tensor_from_spec("9.1", &env).assert_pretty_much_equal_to(&env.scalar(9.1));

        tensor_from_spec("R:[1]", &env).assert_pretty_much_equal_to(&env.create_tensor(vec![Dimension { len: 1, kind: DimensionKind::Regular }], vec![1.0]));
        tensor_from_spec("R-P:[[1,3][5,7]]", &env).assert_pretty_much_equal_to(&env.create_tensor(vec![
            Dimension { len: 2, kind: DimensionKind::Regular },
            Dimension { len: 2, kind: DimensionKind::Polynomial },
        ], vec![1.0, 3.0, 5.0, 7.0]));
        tensor_from_spec("C-R-P:[[[[1,2]][[[3,4]][[5,6]]]", &env).assert_pretty_much_equal_to(&env.create_tensor(vec![
            Dimension { len: 3, kind: DimensionKind::Collection },
            Dimension { len: 1, kind: DimensionKind::Regular },
            Dimension { len: 2, kind: DimensionKind::Polynomial },
        ], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    }
}