use crate::error::{Result, StatError};

/// Result of the Energy Distance test
#[derive(Debug, Clone)]
pub struct EnergyDistanceResult {
    /// The energy distance statistic
    pub statistic: f64,
    /// The p-value (from permutation test)
    pub p_value: f64,
    /// Number of permutations used
    pub n_permutations: usize,
}

/// Compute the Euclidean distance between two points.
#[inline]
fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Compute the energy distance statistic between two samples.
///
/// E(X,Y) = 2*E|X-Y| - E|X-X'| - E|Y-Y'|
/// where X,X' are iid from first distribution, Y,Y' from second.
fn energy_distance_statistic(x: &[&[f64]], y: &[&[f64]]) -> f64 {
    let n = x.len();
    let m = y.len();

    if n == 0 || m == 0 {
        return 0.0;
    }

    let n_f = n as f64;
    let m_f = m as f64;

    // Between-sample distances: E|X-Y|
    let mut sum_xy = 0.0;
    for xi in x.iter() {
        for yj in y.iter() {
            sum_xy += euclidean_distance(xi, yj);
        }
    }
    let mean_xy = sum_xy / (n_f * m_f);

    // Within-sample distances for X: E|X-X'|
    let mut sum_xx = 0.0;
    for i in 0..n {
        for j in 0..n {
            if i != j {
                sum_xx += euclidean_distance(x[i], x[j]);
            }
        }
    }
    let mean_xx = if n > 1 {
        sum_xx / (n_f * (n_f - 1.0))
    } else {
        0.0
    };

    // Within-sample distances for Y: E|Y-Y'|
    let mut sum_yy = 0.0;
    for i in 0..m {
        for j in 0..m {
            if i != j {
                sum_yy += euclidean_distance(y[i], y[j]);
            }
        }
    }
    let mean_yy = if m > 1 {
        sum_yy / (m_f * (m_f - 1.0))
    } else {
        0.0
    };

    // Energy distance
    2.0 * mean_xy - mean_xx - mean_yy
}

/// Perform the Energy Distance test for equality of distributions.
///
/// This is a powerful non-parametric two-sample test that can detect
/// differences in both location and shape of distributions.
///
/// For univariate data, pass single-element slices.
/// For multivariate data, pass vectors of the same dimension.
///
/// # Arguments
/// * `x` - First sample (each element is a d-dimensional observation)
/// * `y` - Second sample (each element is a d-dimensional observation)
/// * `n_permutations` - Number of permutations for p-value estimation
/// * `seed` - Optional random seed for reproducibility
///
/// # Returns
/// * `EnergyDistanceResult` containing energy distance and p-value
///
/// # References
/// * Székely, G.J. and Rizzo, M.L. (2004). "Testing for equal distributions in high dimension"
/// * Székely, G.J. and Rizzo, M.L. (2013). "Energy statistics: A class of statistics based on distances"
pub fn energy_distance_test(
    x: &[Vec<f64>],
    y: &[Vec<f64>],
    n_permutations: usize,
    seed: Option<u64>,
) -> Result<EnergyDistanceResult> {
    if x.is_empty() || y.is_empty() {
        return Err(StatError::EmptyData);
    }

    // Check dimensions match
    let dim = x[0].len();
    if dim == 0 {
        return Err(StatError::InvalidParameter(
            "Data points must have at least one dimension".to_string(),
        ));
    }

    for xi in x.iter() {
        if xi.len() != dim {
            return Err(StatError::InvalidParameter(
                "All data points must have the same dimension".to_string(),
            ));
        }
    }

    for yi in y.iter() {
        if yi.len() != dim {
            return Err(StatError::InvalidParameter(
                "All data points must have the same dimension".to_string(),
            ));
        }
    }

    // Convert to slices for the statistic function
    let x_slices: Vec<&[f64]> = x.iter().map(|v| v.as_slice()).collect();
    let y_slices: Vec<&[f64]> = y.iter().map(|v| v.as_slice()).collect();

    // Compute observed statistic
    let observed = energy_distance_statistic(&x_slices, &y_slices);

    // Combine all observations for permutation testing
    let combined: Vec<Vec<f64>> = x.iter().chain(y.iter()).cloned().collect();
    let n1 = x.len();

    // For multivariate data, we need a custom permutation approach
    use rand::seq::SliceRandom;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    let mut rng = match seed {
        Some(s) => ChaCha8Rng::seed_from_u64(s),
        None => ChaCha8Rng::from_entropy(),
    };

    let mut indices: Vec<usize> = (0..combined.len()).collect();
    let mut count_extreme = 0usize;

    for _ in 0..n_permutations {
        indices.shuffle(&mut rng);

        let perm_x: Vec<&[f64]> = indices[0..n1]
            .iter()
            .map(|&i| combined[i].as_slice())
            .collect();
        let perm_y: Vec<&[f64]> = indices[n1..]
            .iter()
            .map(|&i| combined[i].as_slice())
            .collect();

        let perm_stat = energy_distance_statistic(&perm_x, &perm_y);

        if perm_stat >= observed {
            count_extreme += 1;
        }
    }

    let p_value = (count_extreme as f64 + 1.0) / (n_permutations as f64 + 1.0);

    Ok(EnergyDistanceResult {
        statistic: observed,
        p_value,
        n_permutations,
    })
}

/// Convenience function for univariate energy distance test.
///
/// # Arguments
/// * `x` - First sample (univariate)
/// * `y` - Second sample (univariate)
/// * `n_permutations` - Number of permutations
/// * `seed` - Optional random seed
pub fn energy_distance_test_1d(
    x: &[f64],
    y: &[f64],
    n_permutations: usize,
    seed: Option<u64>,
) -> Result<EnergyDistanceResult> {
    let x_vec: Vec<Vec<f64>> = x.iter().map(|&v| vec![v]).collect();
    let y_vec: Vec<Vec<f64>> = y.iter().map(|&v| vec![v]).collect();
    energy_distance_test(&x_vec, &y_vec, n_permutations, seed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_energy_distance_different_distributions() {
        // Clearly different samples
        let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 1.5, 2.5, 3.5, 4.5, 5.5];
        let y: Vec<f64> = vec![10.0, 11.0, 12.0, 13.0, 14.0, 10.5, 11.5, 12.5, 13.5, 14.5];

        let result = energy_distance_test_1d(&x, &y, 999, Some(42)).unwrap();

        assert!(result.statistic > 0.0);
        assert!(
            result.p_value < 0.05,
            "p_value {} should be < 0.05",
            result.p_value
        );
    }

    #[test]
    fn test_energy_distance_similar_distributions() {
        // Similar samples
        let x: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y: Vec<f64> = vec![1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1];

        let result = energy_distance_test_1d(&x, &y, 999, Some(42)).unwrap();

        // Should not detect significant difference
        assert!(
            result.p_value > 0.1,
            "p_value {} should be > 0.1",
            result.p_value
        );
    }

    #[test]
    fn test_energy_distance_multivariate() {
        // 2D data with clearly different distributions
        let x = vec![
            vec![1.0, 1.0],
            vec![2.0, 2.0],
            vec![3.0, 3.0],
            vec![1.5, 1.5],
            vec![2.5, 2.5],
        ];
        let y = vec![
            vec![10.0, 10.0],
            vec![11.0, 11.0],
            vec![12.0, 12.0],
            vec![10.5, 10.5],
            vec![11.5, 11.5],
        ];

        let result = energy_distance_test(&x, &y, 499, Some(42)).unwrap();

        assert!(
            result.p_value < 0.05,
            "p_value {} should be < 0.05",
            result.p_value
        );
    }

    #[test]
    fn test_energy_distance_empty() {
        let x: Vec<f64> = vec![];
        let y = vec![1.0, 2.0, 3.0];

        assert!(energy_distance_test_1d(&x, &y, 100, None).is_err());
    }
}
