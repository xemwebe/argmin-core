// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Finite Differentiation
//!
//! TODO: Text.

#[cfg(feature = "ndarrayl")]
use ndarray;

pub fn forward_diff_vec_f64(p: &Vec<f64>, op: &Fn(&Vec<f64>) -> f64) -> Vec<f64> {
    let fx = (op)(&p);
    let n = p.len();
    (0..n)
        .map(|i| {
            let mut x1 = p.clone();
            x1[i] += std::f64::EPSILON;
            let fx1 = (op)(&x1);
            (fx1 - fx) / std::f64::EPSILON
        })
        .collect()
}

#[cfg(feature = "ndarrayl")]
pub fn forward_diff_ndarray_f64(
    p: &ndarray::Array1<f64>,
    op: &Fn(&ndarray::Array1<f64>) -> f64,
) -> ndarray::Array1<f64> {
    let fx = (op)(&p);
    let n = p.len();
    (0..n)
        .map(|i| {
            let mut x1 = p.clone();
            x1[i] += std::f64::EPSILON;
            let fx1 = (op)(&x1);
            (fx1 - fx) / std::f64::EPSILON
        })
        .collect()
}

pub fn forward_jacobian_vec_f64(p: &Vec<f64>, op: &Fn(&Vec<f64>) -> Vec<f64>) -> Vec<Vec<f64>> {
    let fx = (op)(&p);
    let n = p.len();
    (0..n)
        .map(|i| {
            let mut x1 = p.clone();
            x1[i] += std::f64::EPSILON;
            let fx1 = (op)(&x1);
            fx1.iter()
                .zip(fx.iter())
                .map(|(a, b)| (a - b) / std::f64::EPSILON)
                .collect::<Vec<f64>>()
        })
        .collect()
}

#[cfg(feature = "ndarrayl")]
pub fn forward_jacobian_ndarray_f64(
    p: &ndarray::Array1<f64>,
    op: &Fn(&ndarray::Array1<f64>) -> ndarray::Array1<f64>,
) -> ndarray::Array2<f64> {
    let fx = (op)(&p);
    let rn = fx.len();
    let n = p.len();
    let mut out = ndarray::Array2::zeros((rn, n));
    for i in 0..n {
        let mut x1 = p.clone();
        x1[i] += std::f64::EPSILON;
        let fx1 = (op)(&x1);
        for j in 0..rn {
            out[(j, i)] = (fx1[j] - fx[j]) / std::f64::EPSILON;
        }
    }
    out
}

pub fn central_diff_vec_f64(p: &Vec<f64>, op: &Fn(&Vec<f64>) -> f64) -> Vec<f64> {
    let n = p.len();
    (0..n)
        .map(|i| {
            let mut x1 = p.clone();
            let mut x2 = p.clone();
            x1[i] += std::f64::EPSILON;
            x2[i] -= std::f64::EPSILON;
            let fx1 = (op)(&x1);
            let fx2 = (op)(&x2);
            (fx1 - fx2) / (2.0 * std::f64::EPSILON)
        })
        .collect()
}

#[cfg(feature = "ndarrayl")]
pub fn central_diff_ndarray_f64(
    p: &ndarray::Array1<f64>,
    op: &Fn(&ndarray::Array1<f64>) -> f64,
) -> ndarray::Array1<f64> {
    let n = p.len();
    (0..n)
        .map(|i| {
            let mut x1 = p.clone();
            let mut x2 = p.clone();
            x1[i] += std::f64::EPSILON;
            x2[i] -= std::f64::EPSILON;
            let fx1 = (op)(&x1);
            let fx2 = (op)(&x2);
            (fx1 - fx2) / (2.0 * std::f64::EPSILON)
        })
        .collect()
}

pub fn central_jacobian_vec_f64(p: &Vec<f64>, op: &Fn(&Vec<f64>) -> Vec<f64>) -> Vec<Vec<f64>> {
    let n = p.len();
    (0..n)
        .map(|i| {
            let mut x1 = p.clone();
            let mut x2 = p.clone();
            x1[i] += std::f64::EPSILON;
            x2[i] -= std::f64::EPSILON;
            let fx1 = (op)(&x1);
            let fx2 = (op)(&x2);
            fx1.iter()
                .zip(fx2.iter())
                .map(|(a, b)| (a - b) / (2.0 * std::f64::EPSILON))
                .collect::<Vec<f64>>()
        })
        .collect()
}

#[cfg(feature = "ndarrayl")]
pub fn central_jacobian_ndarray_f64(
    p: &ndarray::Array1<f64>,
    op: &Fn(&ndarray::Array1<f64>) -> ndarray::Array1<f64>,
) -> ndarray::Array2<f64> {
    let fx = (op)(&p);
    let rn = fx.len();
    let n = p.len();
    let mut out = ndarray::Array2::zeros((rn, n));
    for i in 0..n {
        let mut x1 = p.clone();
        let mut x2 = p.clone();
        x1[i] += std::f64::EPSILON;
        x2[i] -= std::f64::EPSILON;
        let fx1 = (op)(&x1);
        let fx2 = (op)(&x2);
        for j in 0..rn {
            out[(j, i)] = (fx1[j] - fx2[j]) / (2.0 * std::f64::EPSILON);
        }
    }
    out
}

pub trait ArgminFiniteDiff
where
    Self: Sized,
{
    type Jacobian;
    type OperatorOutput;

    fn forward_diff(&self, op: &Fn(&Self) -> f64) -> Self;
    fn forward_jacobian(&self, op: &Fn(&Self) -> Self::OperatorOutput) -> Self::Jacobian;
    fn central_diff(&self, op: &Fn(&Self) -> f64) -> Self;
}

impl ArgminFiniteDiff for Vec<f64>
where
    Self: Sized,
{
    type Jacobian = Vec<Vec<f64>>;
    type OperatorOutput = Vec<f64>;

    fn forward_diff(&self, op: &Fn(&Self) -> f64) -> Self {
        forward_diff_vec_f64(self, op)
    }

    fn central_diff(&self, op: &Fn(&Vec<f64>) -> f64) -> Self {
        central_diff_vec_f64(self, op)
    }

    fn forward_jacobian(&self, op: &Fn(&Self) -> Self::OperatorOutput) -> Self::Jacobian {
        forward_jacobian_vec_f64(self, op)
    }
}

#[cfg(feature = "ndarrayl")]
impl ArgminFiniteDiff for ndarray::Array1<f64>
where
    Self: Sized,
{
    type Jacobian = ndarray::Array2<f64>;
    type OperatorOutput = ndarray::Array1<f64>;

    fn forward_diff(&self, op: &Fn(&Self) -> f64) -> Self {
        forward_diff_ndarray_f64(self, op)
    }

    fn central_diff(&self, op: &Fn(&ndarray::Array1<f64>) -> f64) -> Self {
        central_diff_ndarray_f64(self, op)
    }

    fn forward_jacobian(&self, op: &Fn(&Self) -> Self::OperatorOutput) -> Self::Jacobian {
        forward_jacobian_ndarray_f64(self, op)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ArgminOperator;
    use crate::Error;

    #[test]
    fn test_forward_diff_vec_f64() {
        #[derive(Clone)]
        struct Problem {}

        impl ArgminOperator for Problem {
            type Parameters = Vec<f64>;
            type OperatorOutput = f64;
            type Hessian = ();

            fn apply(&self, p: &Self::Parameters) -> Result<Self::OperatorOutput, Error> {
                Ok(p[0] + p[1].powi(2))
            }

            fn gradient(&self, p: &Self::Parameters) -> Result<Self::Parameters, Error> {
                Ok(p.forward_diff(&|p| self.apply(p).unwrap()))
            }
        }
        let prob = Problem {};
        let p = vec![0.0f64, 1.0f64];
        // println!("{:?}", prob.gradient(&p).unwrap());
        assert!((1.0 - prob.gradient(&p).unwrap()[0]).abs() < std::f64::EPSILON);
        assert!((2.0 - prob.gradient(&p).unwrap()[1]).abs() < std::f64::EPSILON);
    }

    #[cfg(feature = "ndarrayl")]
    #[test]
    fn test_forward_diff_ndarray_f64() {
        use ndarray;

        #[derive(Clone)]
        struct Problem {}

        impl ArgminOperator for Problem {
            type Parameters = ndarray::Array1<f64>;
            type OperatorOutput = f64;
            type Hessian = ();

            fn apply(&self, p: &Self::Parameters) -> Result<Self::OperatorOutput, Error> {
                Ok(p[0] + p[1].powi(2))
            }

            fn gradient(&self, p: &Self::Parameters) -> Result<Self::Parameters, Error> {
                Ok(p.forward_diff(&|p| self.apply(p).unwrap()))
            }
        }
        let prob = Problem {};
        let p = ndarray::Array1::from_vec(vec![0.0f64, 1.0f64]);
        // println!("{:?}", prob.gradient(&p).unwrap());
        assert!((1.0 - prob.gradient(&p).unwrap()[0]).abs() < std::f64::EPSILON);
        assert!((2.0 - prob.gradient(&p).unwrap()[1]).abs() < std::f64::EPSILON);
    }

    #[test]
    fn test_central_diff_vec_f64() {
        #[derive(Clone)]
        struct Problem {}

        impl ArgminOperator for Problem {
            type Parameters = Vec<f64>;
            type OperatorOutput = f64;
            type Hessian = ();

            fn apply(&self, p: &Self::Parameters) -> Result<Self::OperatorOutput, Error> {
                Ok(p[0] + p[1].powi(2))
            }

            fn gradient(&self, p: &Self::Parameters) -> Result<Self::Parameters, Error> {
                Ok(p.central_diff(&|p| self.apply(p).unwrap()))
            }
        }
        let prob = Problem {};
        let p = vec![0.0f64, 1.0f64];
        // println!("{:?}", prob.gradient(&p).unwrap());
        assert!((1.0 - prob.gradient(&p).unwrap()[0]).abs() < std::f64::EPSILON);
        assert!((2.0 - prob.gradient(&p).unwrap()[1]).abs() < std::f64::EPSILON);
    }

    #[test]
    fn test_central_diff_ndarray_f64() {
        #[derive(Clone)]
        struct Problem {}

        impl ArgminOperator for Problem {
            type Parameters = Vec<f64>;
            type OperatorOutput = f64;
            type Hessian = ();

            fn apply(&self, p: &Self::Parameters) -> Result<Self::OperatorOutput, Error> {
                Ok(p[0] + p[1].powi(2))
            }

            fn gradient(&self, p: &Self::Parameters) -> Result<Self::Parameters, Error> {
                Ok(p.central_diff(&|p| self.apply(p).unwrap()))
            }
        }
        let prob = Problem {};
        let p = vec![0.0f64, 1.0f64];
        // println!("{:?}", prob.gradient(&p).unwrap());
        assert!((1.0 - prob.gradient(&p).unwrap()[0]).abs() < std::f64::EPSILON);
        assert!((2.0 - prob.gradient(&p).unwrap()[1]).abs() < std::f64::EPSILON);
    }

    #[test]
    fn test_forward_jacobian_vec_f64() {
        let op = |x: &Vec<f64>| {
            vec![
                2.0 * (x[1].powi(3) - x[0].powi(2)),
                3.0 * (x[1].powi(3) - x[0].powi(2)) + 2.0 * (x[2].powi(3) - x[1].powi(2)),
                3.0 * (x[2].powi(3) - x[1].powi(2)) + 2.0 * (x[3].powi(3) - x[2].powi(2)),
                3.0 * (x[3].powi(3) - x[2].powi(2)) + 2.0 * (x[4].powi(3) - x[3].powi(2)),
                3.0 * (x[4].powi(3) - x[3].powi(2)) + 2.0 * (x[5].powi(3) - x[4].powi(2)),
                3.0 * (x[5].powi(3) - x[4].powi(2)),
            ]
        };
        let p = vec![1.0f64, 1.0, 1.0, 1.0, 1.0, 1.0];
        let jacobian = forward_jacobian_vec_f64(&p, &op);
        let res = vec![
            vec![-4.0, -6.0, 0.0, 0.0, 0.0, 0.0],
            vec![6.0, 5.0, -6.0, 0.0, 0.0, 0.0],
            vec![0.0, 6.0, 5.0, -6.0, 0.0, 0.0],
            vec![0.0, 0.0, 6.0, 5.0, -6.0, 0.0],
            vec![0.0, 0.0, 0.0, 6.0, 5.0, -6.0],
            vec![0.0, 0.0, 0.0, 0.0, 6.0, 9.0],
        ];
        // println!("{:?}", jacobian);
        res.iter()
            .zip(jacobian.iter())
            .map(|(r, j)| {
                r.iter()
                    .zip(j.iter())
                    .map(|(a, b)| assert!((a - b).abs() < std::f64::EPSILON))
            })
            .count();
    }

    #[test]
    fn test_forward_jacobian_vec_f64_trait() {
        let op = |x: &Vec<f64>| {
            vec![
                2.0 * (x[1].powi(3) - x[0].powi(2)),
                3.0 * (x[1].powi(3) - x[0].powi(2)) + 2.0 * (x[2].powi(3) - x[1].powi(2)),
                3.0 * (x[2].powi(3) - x[1].powi(2)) + 2.0 * (x[3].powi(3) - x[2].powi(2)),
                3.0 * (x[3].powi(3) - x[2].powi(2)) + 2.0 * (x[4].powi(3) - x[3].powi(2)),
                3.0 * (x[4].powi(3) - x[3].powi(2)) + 2.0 * (x[5].powi(3) - x[4].powi(2)),
                3.0 * (x[5].powi(3) - x[4].powi(2)),
            ]
        };
        let p = vec![1.0f64, 1.0, 1.0, 1.0, 1.0, 1.0];
        let jacobian = p.forward_jacobian(&op);
        let res = vec![
            vec![-4.0, -6.0, 0.0, 0.0, 0.0, 0.0],
            vec![6.0, 5.0, -6.0, 0.0, 0.0, 0.0],
            vec![0.0, 6.0, 5.0, -6.0, 0.0, 0.0],
            vec![0.0, 0.0, 6.0, 5.0, -6.0, 0.0],
            vec![0.0, 0.0, 0.0, 6.0, 5.0, -6.0],
            vec![0.0, 0.0, 0.0, 0.0, 6.0, 9.0],
        ];
        // println!("{:?}", jacobian);
        res.iter()
            .zip(jacobian.iter())
            .map(|(r, j)| {
                r.iter()
                    .zip(j.iter())
                    .map(|(a, b)| assert!((a - b).abs() < std::f64::EPSILON))
            })
            .count();
    }

    #[cfg(feature = "ndarrayl")]
    #[test]
    fn test_forward_jacobian_ndarray_f64() {
        let op = |x: &ndarray::Array1<f64>| {
            ndarray::Array1::from_vec(vec![
                2.0 * (x[1].powi(3) - x[0].powi(2)),
                3.0 * (x[1].powi(3) - x[0].powi(2)) + 2.0 * (x[2].powi(3) - x[1].powi(2)),
                3.0 * (x[2].powi(3) - x[1].powi(2)) + 2.0 * (x[3].powi(3) - x[2].powi(2)),
                3.0 * (x[3].powi(3) - x[2].powi(2)) + 2.0 * (x[4].powi(3) - x[3].powi(2)),
                3.0 * (x[4].powi(3) - x[3].powi(2)) + 2.0 * (x[5].powi(3) - x[4].powi(2)),
                3.0 * (x[5].powi(3) - x[4].powi(2)),
            ])
        };
        let p = ndarray::Array1::from_vec(vec![1.0f64, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let jacobian = forward_jacobian_ndarray_f64(&p, &op);
        let res = vec![
            vec![-4.0, -6.0, 0.0, 0.0, 0.0, 0.0],
            vec![6.0, 5.0, -6.0, 0.0, 0.0, 0.0],
            vec![0.0, 6.0, 5.0, -6.0, 0.0, 0.0],
            vec![0.0, 0.0, 6.0, 5.0, -6.0, 0.0],
            vec![0.0, 0.0, 0.0, 6.0, 5.0, -6.0],
            vec![0.0, 0.0, 0.0, 0.0, 6.0, 9.0],
        ];
        // println!("{:?}", jacobian);
        (0..6)
            .zip(0..6)
            .map(|(i, j)| assert!((res[i][j] - jacobian[(i, j)]).abs() < std::f64::EPSILON))
            .count();
    }

    #[cfg(feature = "ndarrayl")]
    #[test]
    fn test_forward_jacobian_ndarray_f64_trait() {
        let op = |x: &ndarray::Array1<f64>| {
            ndarray::Array1::from_vec(vec![
                2.0 * (x[1].powi(3) - x[0].powi(2)),
                3.0 * (x[1].powi(3) - x[0].powi(2)) + 2.0 * (x[2].powi(3) - x[1].powi(2)),
                3.0 * (x[2].powi(3) - x[1].powi(2)) + 2.0 * (x[3].powi(3) - x[2].powi(2)),
                3.0 * (x[3].powi(3) - x[2].powi(2)) + 2.0 * (x[4].powi(3) - x[3].powi(2)),
                3.0 * (x[4].powi(3) - x[3].powi(2)) + 2.0 * (x[5].powi(3) - x[4].powi(2)),
                3.0 * (x[5].powi(3) - x[4].powi(2)),
            ])
        };
        let p = ndarray::Array1::from_vec(vec![1.0f64, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let jacobian = p.forward_jacobian(&op);
        let res = vec![
            vec![-4.0, -6.0, 0.0, 0.0, 0.0, 0.0],
            vec![6.0, 5.0, -6.0, 0.0, 0.0, 0.0],
            vec![0.0, 6.0, 5.0, -6.0, 0.0, 0.0],
            vec![0.0, 0.0, 6.0, 5.0, -6.0, 0.0],
            vec![0.0, 0.0, 0.0, 6.0, 5.0, -6.0],
            vec![0.0, 0.0, 0.0, 0.0, 6.0, 9.0],
        ];
        // println!("{:?}", jacobian);
        (0..6)
            .zip(0..6)
            .map(|(i, j)| assert!((res[i][j] - jacobian[(i, j)]).abs() < std::f64::EPSILON))
            .count();
    }

    #[test]
    fn test_central_jacobian_vec_f64() {
        let op = |x: &Vec<f64>| {
            vec![
                2.0 * (x[1].powi(3) - x[0].powi(2)),
                3.0 * (x[1].powi(3) - x[0].powi(2)) + 2.0 * (x[2].powi(3) - x[1].powi(2)),
                3.0 * (x[2].powi(3) - x[1].powi(2)) + 2.0 * (x[3].powi(3) - x[2].powi(2)),
                3.0 * (x[3].powi(3) - x[2].powi(2)) + 2.0 * (x[4].powi(3) - x[3].powi(2)),
                3.0 * (x[4].powi(3) - x[3].powi(2)) + 2.0 * (x[5].powi(3) - x[4].powi(2)),
                3.0 * (x[5].powi(3) - x[4].powi(2)),
            ]
        };
        let p = vec![1.0f64, 1.0, 1.0, 1.0, 1.0, 1.0];
        let jacobian = central_jacobian_vec_f64(&p, &op);
        let res = vec![
            vec![-4.0, -6.0, 0.0, 0.0, 0.0, 0.0],
            vec![6.0, 5.0, -6.0, 0.0, 0.0, 0.0],
            vec![0.0, 6.0, 5.0, -6.0, 0.0, 0.0],
            vec![0.0, 0.0, 6.0, 5.0, -6.0, 0.0],
            vec![0.0, 0.0, 0.0, 6.0, 5.0, -6.0],
            vec![0.0, 0.0, 0.0, 0.0, 6.0, 9.0],
        ];
        // println!("{:?}", jacobian);
        res.iter()
            .zip(jacobian.iter())
            .map(|(r, j)| {
                r.iter()
                    .zip(j.iter())
                    .map(|(a, b)| assert!((a - b).abs() < std::f64::EPSILON))
            })
            .count();
    }

    #[cfg(feature = "ndarrayl")]
    #[test]
    fn test_central_jacobian_ndarray_f64() {
        let op = |x: &ndarray::Array1<f64>| {
            ndarray::Array1::from_vec(vec![
                2.0 * (x[1].powi(3) - x[0].powi(2)),
                3.0 * (x[1].powi(3) - x[0].powi(2)) + 2.0 * (x[2].powi(3) - x[1].powi(2)),
                3.0 * (x[2].powi(3) - x[1].powi(2)) + 2.0 * (x[3].powi(3) - x[2].powi(2)),
                3.0 * (x[3].powi(3) - x[2].powi(2)) + 2.0 * (x[4].powi(3) - x[3].powi(2)),
                3.0 * (x[4].powi(3) - x[3].powi(2)) + 2.0 * (x[5].powi(3) - x[4].powi(2)),
                3.0 * (x[5].powi(3) - x[4].powi(2)),
            ])
        };
        let p = ndarray::Array1::from_vec(vec![1.0f64, 1.0, 1.0, 1.0, 1.0, 1.0]);
        let jacobian = central_jacobian_ndarray_f64(&p, &op);
        let res = vec![
            vec![-4.0, -6.0, 0.0, 0.0, 0.0, 0.0],
            vec![6.0, 5.0, -6.0, 0.0, 0.0, 0.0],
            vec![0.0, 6.0, 5.0, -6.0, 0.0, 0.0],
            vec![0.0, 0.0, 6.0, 5.0, -6.0, 0.0],
            vec![0.0, 0.0, 0.0, 6.0, 5.0, -6.0],
            vec![0.0, 0.0, 0.0, 0.0, 6.0, 9.0],
        ];
        // println!("{:?}", jacobian);
        (0..6)
            .zip(0..6)
            .map(|(i, j)| assert!((res[i][j] - jacobian[(i, j)]).abs() < std::f64::EPSILON))
            .count();
    }
}
