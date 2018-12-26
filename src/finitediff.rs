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

const EPS_F64: f64 = 4.0 * std::f64::EPSILON;

pub struct PerturbationVector {
    pub x_idx: Vec<usize>,
    pub r_idx: Vec<Vec<usize>>,
}

impl PerturbationVector {
    pub fn new() -> Self {
        PerturbationVector {
            x_idx: vec![],
            r_idx: vec![],
        }
    }

    pub fn add(mut self, x_idx: usize, r_idx: Vec<usize>) -> Self {
        self.x_idx.push(x_idx);
        self.r_idx.push(r_idx);
        self
    }
}

pub type PerturbationVectors = Vec<PerturbationVector>;

pub fn forward_diff_vec_f64(p: &Vec<f64>, op: &Fn(&Vec<f64>) -> f64) -> Vec<f64> {
    let fx = (op)(&p);
    let n = p.len();
    (0..n)
        .map(|i| {
            let mut x1 = p.clone();
            x1[i] += EPS_F64.sqrt();
            let fx1 = (op)(&x1);
            (fx1 - fx) / (EPS_F64.sqrt())
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
            x1[i] += EPS_F64.sqrt();
            let fx1 = (op)(&x1);
            (fx1 - fx) / (EPS_F64.sqrt())
        })
        .collect()
}

pub fn forward_jacobian_vec_f64(p: &Vec<f64>, op: &Fn(&Vec<f64>) -> Vec<f64>) -> Vec<Vec<f64>> {
    let fx = (op)(&p);
    let n = p.len();
    (0..n)
        .map(|i| {
            let mut x1 = p.clone();
            x1[i] += EPS_F64.sqrt();
            let fx1 = (op)(&x1);
            fx1.iter()
                .zip(fx.iter())
                .map(|(a, b)| (a - b) / EPS_F64.sqrt())
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
        x1[i] += EPS_F64.sqrt();
        let fx1 = (op)(&x1);
        for j in 0..rn {
            out[(j, i)] = (fx1[j] - fx[j]) / EPS_F64.sqrt();
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
            x1[i] += EPS_F64.sqrt();
            x2[i] -= EPS_F64.sqrt();
            let fx1 = (op)(&x1);
            let fx2 = (op)(&x2);
            (fx1 - fx2) / (2.0 * EPS_F64.sqrt())
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
            x1[i] += EPS_F64.sqrt();
            x2[i] -= EPS_F64.sqrt();
            let fx1 = (op)(&x1);
            let fx2 = (op)(&x2);
            (fx1 - fx2) / (2.0 * EPS_F64.sqrt())
        })
        .collect()
}

pub fn central_jacobian_vec_f64(p: &Vec<f64>, op: &Fn(&Vec<f64>) -> Vec<f64>) -> Vec<Vec<f64>> {
    let n = p.len();
    (0..n)
        .map(|i| {
            let mut x1 = p.clone();
            let mut x2 = p.clone();
            x1[i] += EPS_F64.sqrt();
            x2[i] -= EPS_F64.sqrt();
            let fx1 = (op)(&x1);
            let fx2 = (op)(&x2);
            fx1.iter()
                .zip(fx2.iter())
                .map(|(a, b)| (a - b) / (2.0 * EPS_F64.sqrt()))
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
        x1[i] += EPS_F64.sqrt();
        x2[i] -= EPS_F64.sqrt();
        let fx1 = (op)(&x1);
        let fx2 = (op)(&x2);
        for j in 0..rn {
            out[(j, i)] = (fx1[j] - fx2[j]) / (2.0 * EPS_F64.sqrt());
        }
    }
    out
}

pub fn forward_jacobian_pert_vec_f64(
    p: &Vec<f64>,
    op: &Fn(&Vec<f64>) -> Vec<f64>,
    pert: PerturbationVectors,
) -> Vec<Vec<f64>> {
    let fx = (op)(&p);
    let n = pert.len();
    let mut out = vec![vec![0.0; p.len()]; fx.len()];
    for i in 0..n {
        let mut x1 = p.clone();
        for j in pert[i].x_idx.iter() {
            x1[*j] += EPS_F64.sqrt();
        }
        let fx1 = (op)(&x1);
        for (k, x_idx) in pert[i].x_idx.iter().enumerate() {
            for j in pert[i].r_idx[k].iter() {
                out[*x_idx][*j] = (fx1[*j] - fx[*j]) / EPS_F64.sqrt();
            }
        }
    }
    out
}

#[cfg(feature = "ndarrayl")]
pub fn forward_jacobian_pert_ndarray_f64(
    p: &ndarray::Array1<f64>,
    op: &Fn(&ndarray::Array1<f64>) -> ndarray::Array1<f64>,
    pert: PerturbationVectors,
) -> ndarray::Array2<f64> {
    let fx = (op)(&p);
    let n = pert.len();
    let mut out = ndarray::Array2::zeros((fx.len(), p.len()));
    for i in 0..n {
        let mut x1 = p.clone();
        for j in pert[i].x_idx.iter() {
            x1[*j] += EPS_F64.sqrt();
        }
        let fx1 = (op)(&x1);
        for (k, x_idx) in pert[i].x_idx.iter().enumerate() {
            for j in pert[i].r_idx[k].iter() {
                out[(*x_idx, *j)] = (fx1[*j] - fx[*j]) / EPS_F64.sqrt();
            }
        }
    }
    out
}

pub fn central_jacobian_pert_vec_f64(
    p: &Vec<f64>,
    op: &Fn(&Vec<f64>) -> Vec<f64>,
    pert: PerturbationVectors,
) -> Vec<Vec<f64>> {
    let n = pert.len();
    let mut out = vec![];
    for i in 0..n {
        let mut x1 = p.clone();
        let mut x2 = p.clone();
        for j in pert[i].x_idx.iter() {
            x1[*j] += EPS_F64.sqrt();
            x2[*j] -= EPS_F64.sqrt();
        }
        let fx1 = (op)(&x1);
        let fx2 = (op)(&x2);
        if i == 0 {
            out = vec![vec![0.0; p.len()]; fx1.len()];
        }
        for (k, x_idx) in pert[i].x_idx.iter().enumerate() {
            for j in pert[i].r_idx[k].iter() {
                out[*x_idx][*j] = (fx1[*j] - fx2[*j]) / (2.0 * EPS_F64.sqrt());
            }
        }
    }
    out
}

#[cfg(feature = "ndarrayl")]
pub fn central_jacobian_pert_ndarray_f64(
    p: &ndarray::Array1<f64>,
    op: &Fn(&ndarray::Array1<f64>) -> ndarray::Array1<f64>,
    pert: PerturbationVectors,
) -> ndarray::Array2<f64> {
    let n = pert.len();
    let mut out = ndarray::Array2::zeros((1, 1));
    for i in 0..n {
        let mut x1 = p.clone();
        let mut x2 = p.clone();
        for j in pert[i].x_idx.iter() {
            x1[*j] += EPS_F64.sqrt();
            x2[*j] -= EPS_F64.sqrt();
        }
        let fx1 = (op)(&x1);
        let fx2 = (op)(&x2);
        if i == 0 {
            out = ndarray::Array2::zeros((fx1.len(), p.len()));
        }
        for (k, x_idx) in pert[i].x_idx.iter().enumerate() {
            for j in pert[i].r_idx[k].iter() {
                out[(*x_idx, *j)] = (fx1[*j] - fx2[*j]) / (2.0 * EPS_F64.sqrt());
            }
        }
    }
    out
}

pub fn forward_hessian_vec_f64(p: &Vec<f64>, grad: &Fn(&Vec<f64>) -> Vec<f64>) -> Vec<Vec<f64>> {
    let fx = (grad)(p);
    let n = p.len();
    let mut out: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut x1 = p.clone();
            x1[i] += EPS_F64.sqrt();
            let fx1 = (grad)(&x1);
            fx1.iter()
                .zip(fx.iter())
                .map(|(a, b)| (a - b) / (EPS_F64.sqrt()))
                .collect::<Vec<f64>>()
        })
        .collect();

    // restore symmetry
    for i in 0..n {
        for j in (i + 1)..n {
            let t = (out[i][j] + out[j][i]) / 2.0;
            out[i][j] = t;
            out[j][i] = t;
        }
    }
    out
}

#[cfg(feature = "ndarrayl")]
pub fn forward_hessian_ndarray_f64(
    p: &ndarray::Array1<f64>,
    grad: &Fn(&ndarray::Array1<f64>) -> ndarray::Array1<f64>,
) -> ndarray::Array2<f64> {
    let fx = (grad)(&p);
    let rn = fx.len();
    let n = p.len();
    let mut out = ndarray::Array2::zeros((rn, n));
    for i in 0..n {
        let mut x1 = p.clone();
        x1[i] += EPS_F64.sqrt();
        let fx1 = (grad)(&x1);
        for j in 0..rn {
            out[(j, i)] = (fx1[j] - fx[j]) / EPS_F64.sqrt();
        }
    }
    // restore symmetry
    for i in 0..n {
        for j in (i + 1)..n {
            let t = (out[(i, j)] + out[(j, i)]) / 2.0;
            out[(i, j)] = t;
            out[(j, i)] = t;
        }
    }
    out
}

pub fn central_hessian_vec_f64(p: &Vec<f64>, grad: &Fn(&Vec<f64>) -> Vec<f64>) -> Vec<Vec<f64>> {
    // let fx = (grad)(p);
    let n = p.len();
    let mut out: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut x1 = p.clone();
            let mut x2 = p.clone();
            x1[i] += EPS_F64.sqrt();
            x2[i] -= EPS_F64.sqrt();
            let fx1 = (grad)(&x1);
            let fx2 = (grad)(&x2);
            fx1.iter()
                .zip(fx2.iter())
                .map(|(a, b)| (a - b) / (2.0 * EPS_F64.sqrt()))
                .collect::<Vec<f64>>()
        })
        .collect();

    // restore symmetry
    for i in 0..n {
        for j in (i + 1)..n {
            let t = (out[i][j] + out[j][i]) / 2.0;
            out[i][j] = t;
            out[j][i] = t;
        }
    }
    out
}

#[cfg(feature = "ndarrayl")]
pub fn central_hessian_ndarray_f64(
    p: &ndarray::Array1<f64>,
    grad: &Fn(&ndarray::Array1<f64>) -> ndarray::Array1<f64>,
) -> ndarray::Array2<f64> {
    let fx = (grad)(&p);
    let rn = fx.len();
    let n = p.len();
    let mut out = ndarray::Array2::zeros((rn, n));
    for i in 0..n {
        let mut x1 = p.clone();
        let mut x2 = p.clone();
        x1[i] += EPS_F64.sqrt();
        x2[i] -= EPS_F64.sqrt();
        let fx1 = (grad)(&x1);
        let fx2 = (grad)(&x2);
        for j in 0..rn {
            out[(j, i)] = (fx1[j] - fx2[j]) / (2.0 * EPS_F64.sqrt());
        }
    }
    // restore symmetry
    for i in 0..n {
        for j in (i + 1)..n {
            let t = (out[(i, j)] + out[(j, i)]) / 2.0;
            out[(i, j)] = t;
            out[(j, i)] = t;
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
    fn central_diff(&self, op: &Fn(&Self) -> f64) -> Self;
    fn forward_jacobian(&self, op: &Fn(&Self) -> Self::OperatorOutput) -> Self::Jacobian;
    fn central_jacobian(&self, op: &Fn(&Self) -> Self::OperatorOutput) -> Self::Jacobian;
    fn forward_jacobian_pert(
        &self,
        op: &Fn(&Self) -> Self::OperatorOutput,
        pert: PerturbationVectors,
    ) -> Self::Jacobian;
    fn central_jacobian_pert(
        &self,
        op: &Fn(&Self) -> Self::OperatorOutput,
        pert: PerturbationVectors,
    ) -> Self::Jacobian;
    fn forward_hessian(&self, grad: &Fn(&Self) -> Self::OperatorOutput) -> Self::Jacobian;
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

    fn central_jacobian(&self, op: &Fn(&Self) -> Self::OperatorOutput) -> Self::Jacobian {
        central_jacobian_vec_f64(self, op)
    }

    fn forward_jacobian_pert(
        &self,
        op: &Fn(&Self) -> Self::OperatorOutput,
        pert: PerturbationVectors,
    ) -> Self::Jacobian {
        forward_jacobian_pert_vec_f64(self, op, pert)
    }

    fn central_jacobian_pert(
        &self,
        op: &Fn(&Self) -> Self::OperatorOutput,
        pert: PerturbationVectors,
    ) -> Self::Jacobian {
        central_jacobian_pert_vec_f64(self, op, pert)
    }

    fn forward_hessian(&self, grad: &Fn(&Self) -> Self::OperatorOutput) -> Self::Jacobian {
        forward_hessian_vec_f64(self, grad)
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

    fn central_jacobian(&self, op: &Fn(&Self) -> Self::OperatorOutput) -> Self::Jacobian {
        central_jacobian_ndarray_f64(self, op)
    }

    fn forward_jacobian_pert(
        &self,
        op: &Fn(&Self) -> Self::OperatorOutput,
        pert: PerturbationVectors,
    ) -> Self::Jacobian {
        forward_jacobian_pert_ndarray_f64(self, op, pert)
    }

    fn central_jacobian_pert(
        &self,
        op: &Fn(&Self) -> Self::OperatorOutput,
        pert: PerturbationVectors,
    ) -> Self::Jacobian {
        central_jacobian_pert_ndarray_f64(self, op, pert)
    }

    fn forward_hessian(&self, grad: &Fn(&Self) -> Self::OperatorOutput) -> Self::Jacobian {
        forward_hessian_ndarray_f64(self, grad)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ArgminOperator;
    use crate::Error;

    const COMP_ACC: f64 = 1e-6;

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
        let p = vec![1.0f64, 1.0f64];
        assert!((1.0 - prob.gradient(&p).unwrap()[0]).abs() < COMP_ACC);
        assert!((2.0 - prob.gradient(&p).unwrap()[1]).abs() < COMP_ACC);
        let p = vec![1.0f64, 2.0f64];
        assert!((1.0 - prob.gradient(&p).unwrap()[0]).abs() < COMP_ACC);
        assert!((4.0 - prob.gradient(&p).unwrap()[1]).abs() < COMP_ACC);
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
        let p = ndarray::Array1::from_vec(vec![1.0f64, 1.0f64]);
        // println!("{:?}", prob.gradient(&p).unwrap());
        assert!((1.0 - prob.gradient(&p).unwrap()[0]).abs() < COMP_ACC);
        assert!((2.0 - prob.gradient(&p).unwrap()[1]).abs() < COMP_ACC);
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
        let p = vec![1.0f64, 1.0f64];
        // println!("{:?}", prob.gradient(&p).unwrap());
        assert!((1.0 - prob.gradient(&p).unwrap()[0]).abs() < COMP_ACC);
        assert!((2.0 - prob.gradient(&p).unwrap()[1]).abs() < COMP_ACC);
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
        let p = vec![1.0f64, 1.0f64];
        // println!("{:?}", prob.gradient(&p).unwrap());
        assert!((1.0 - prob.gradient(&p).unwrap()[0]).abs() < COMP_ACC);
        assert!((2.0 - prob.gradient(&p).unwrap()[1]).abs() < COMP_ACC);
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
        (0..6)
            .zip(0..6)
            .map(|(i, j)| assert!((res[i][j] - jacobian[i][j]).abs() < COMP_ACC))
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
        (0..6)
            .zip(0..6)
            .map(|(i, j)| assert!((res[i][j] - jacobian[i][j]).abs() < COMP_ACC))
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
            .map(|(i, j)| assert!((res[i][j] - jacobian[(i, j)]).abs() < COMP_ACC))
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
            .map(|(i, j)| assert!((res[i][j] - jacobian[(i, j)]).abs() < COMP_ACC))
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
        (0..6)
            .zip(0..6)
            .map(|(i, j)| assert!((res[i][j] - jacobian[i][j]).abs() < COMP_ACC))
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
            .map(|(i, j)| assert!((res[i][j] - jacobian[(i, j)]).abs() < COMP_ACC))
            .count();
    }

    #[test]
    fn test_central_jacobian_vec_f64_trait() {
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
        let jacobian = p.central_jacobian(&op);
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
            .map(|(i, j)| assert!((res[i][j] - jacobian[i][j]).abs() < COMP_ACC))
            .count();
    }

    #[cfg(feature = "ndarrayl")]
    #[test]
    fn test_central_jacobian_ndarray_f64_trait() {
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
        let jacobian = p.central_jacobian(&op);
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
            .map(|(i, j)| assert!((res[i][j] - jacobian[(i, j)]).abs() < COMP_ACC))
            .count();
    }

    #[test]
    fn test_forward_jacobian_pert_vec_f64() {
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
        let pert = vec![
            PerturbationVector::new()
                .add(0, vec![0, 1])
                .add(3, vec![2, 3, 4]),
            PerturbationVector::new()
                .add(1, vec![0, 1, 2])
                .add(4, vec![3, 4, 5]),
            PerturbationVector::new()
                .add(2, vec![1, 2, 3])
                .add(5, vec![4, 5]),
        ];
        let jacobian = forward_jacobian_pert_vec_f64(&p, &op, pert);
        let res = vec![
            vec![-4.0, -6.0, 0.0, 0.0, 0.0, 0.0],
            vec![6.0, 5.0, -6.0, 0.0, 0.0, 0.0],
            vec![0.0, 6.0, 5.0, -6.0, 0.0, 0.0],
            vec![0.0, 0.0, 6.0, 5.0, -6.0, 0.0],
            vec![0.0, 0.0, 0.0, 6.0, 5.0, -6.0],
            vec![0.0, 0.0, 0.0, 0.0, 6.0, 9.0],
        ];
        // println!("jacobian:\n{:?}", jacobian);
        // println!("res:\n{:?}", res);
        (0..6)
            .zip(0..6)
            .map(|(i, j)| assert!((res[i][j] - jacobian[i][j]).abs() < COMP_ACC))
            .count();
    }

    #[cfg(feature = "ndarrayl")]
    #[test]
    fn test_forward_jacobian_pert_ndarray_f64() {
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
        let pert = vec![
            PerturbationVector::new()
                .add(0, vec![0, 1])
                .add(3, vec![2, 3, 4]),
            PerturbationVector::new()
                .add(1, vec![0, 1, 2])
                .add(4, vec![3, 4, 5]),
            PerturbationVector::new()
                .add(2, vec![1, 2, 3])
                .add(5, vec![4, 5]),
        ];
        let jacobian = forward_jacobian_pert_ndarray_f64(&p, &op, pert);
        let res = vec![
            vec![-4.0, -6.0, 0.0, 0.0, 0.0, 0.0],
            vec![6.0, 5.0, -6.0, 0.0, 0.0, 0.0],
            vec![0.0, 6.0, 5.0, -6.0, 0.0, 0.0],
            vec![0.0, 0.0, 6.0, 5.0, -6.0, 0.0],
            vec![0.0, 0.0, 0.0, 6.0, 5.0, -6.0],
            vec![0.0, 0.0, 0.0, 0.0, 6.0, 9.0],
        ];
        // println!("jacobian:\n{:?}", jacobian);
        // println!("res:\n{:?}", res);
        (0..6)
            .zip(0..6)
            .map(|(i, j)| assert!((res[i][j] - jacobian[(i, j)]).abs() < COMP_ACC))
            .count();
    }

    #[test]
    fn test_forward_jacobian_pert_vec_f64_trait() {
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
        let pert = vec![
            PerturbationVector::new()
                .add(0, vec![0, 1])
                .add(3, vec![2, 3, 4]),
            PerturbationVector::new()
                .add(1, vec![0, 1, 2])
                .add(4, vec![3, 4, 5]),
            PerturbationVector::new()
                .add(2, vec![1, 2, 3])
                .add(5, vec![4, 5]),
        ];
        let jacobian = p.forward_jacobian_pert(&op, pert);
        let res = vec![
            vec![-4.0, -6.0, 0.0, 0.0, 0.0, 0.0],
            vec![6.0, 5.0, -6.0, 0.0, 0.0, 0.0],
            vec![0.0, 6.0, 5.0, -6.0, 0.0, 0.0],
            vec![0.0, 0.0, 6.0, 5.0, -6.0, 0.0],
            vec![0.0, 0.0, 0.0, 6.0, 5.0, -6.0],
            vec![0.0, 0.0, 0.0, 0.0, 6.0, 9.0],
        ];
        // println!("jacobian:\n{:?}", jacobian);
        // println!("res:\n{:?}", res);
        (0..6)
            .zip(0..6)
            .map(|(i, j)| assert!((res[i][j] - jacobian[i][j]).abs() < COMP_ACC))
            .count();
    }

    #[cfg(feature = "ndarrayl")]
    #[test]
    fn test_forward_jacobian_pert_ndarray_f64_trait() {
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
        let pert = vec![
            PerturbationVector::new()
                .add(0, vec![0, 1])
                .add(3, vec![2, 3, 4]),
            PerturbationVector::new()
                .add(1, vec![0, 1, 2])
                .add(4, vec![3, 4, 5]),
            PerturbationVector::new()
                .add(2, vec![1, 2, 3])
                .add(5, vec![4, 5]),
        ];
        let jacobian = p.forward_jacobian_pert(&op, pert);
        let res = vec![
            vec![-4.0, -6.0, 0.0, 0.0, 0.0, 0.0],
            vec![6.0, 5.0, -6.0, 0.0, 0.0, 0.0],
            vec![0.0, 6.0, 5.0, -6.0, 0.0, 0.0],
            vec![0.0, 0.0, 6.0, 5.0, -6.0, 0.0],
            vec![0.0, 0.0, 0.0, 6.0, 5.0, -6.0],
            vec![0.0, 0.0, 0.0, 0.0, 6.0, 9.0],
        ];
        // println!("jacobian:\n{:?}", jacobian);
        // println!("res:\n{:?}", res);
        (0..6)
            .zip(0..6)
            .map(|(i, j)| assert!((res[i][j] - jacobian[(i, j)]).abs() < COMP_ACC))
            .count();
    }

    #[test]
    fn test_central_jacobian_pert_vec_f64() {
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
        let pert = vec![
            PerturbationVector::new()
                .add(0, vec![0, 1])
                .add(3, vec![2, 3, 4]),
            PerturbationVector::new()
                .add(1, vec![0, 1, 2])
                .add(4, vec![3, 4, 5]),
            PerturbationVector::new()
                .add(2, vec![1, 2, 3])
                .add(5, vec![4, 5]),
        ];
        let jacobian = central_jacobian_pert_vec_f64(&p, &op, pert);
        let res = vec![
            vec![-4.0, -6.0, 0.0, 0.0, 0.0, 0.0],
            vec![6.0, 5.0, -6.0, 0.0, 0.0, 0.0],
            vec![0.0, 6.0, 5.0, -6.0, 0.0, 0.0],
            vec![0.0, 0.0, 6.0, 5.0, -6.0, 0.0],
            vec![0.0, 0.0, 0.0, 6.0, 5.0, -6.0],
            vec![0.0, 0.0, 0.0, 0.0, 6.0, 9.0],
        ];
        // println!("jacobian:\n{:?}", jacobian);
        // println!("res:\n{:?}", res);
        (0..6)
            .zip(0..6)
            .map(|(i, j)| assert!((res[i][j] - jacobian[i][j]).abs() < COMP_ACC))
            .count();
    }

    #[cfg(feature = "ndarrayl")]
    #[test]
    fn test_central_jacobian_pert_ndarray_f64() {
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
        let pert = vec![
            PerturbationVector::new()
                .add(0, vec![0, 1])
                .add(3, vec![2, 3, 4]),
            PerturbationVector::new()
                .add(1, vec![0, 1, 2])
                .add(4, vec![3, 4, 5]),
            PerturbationVector::new()
                .add(2, vec![1, 2, 3])
                .add(5, vec![4, 5]),
        ];
        let jacobian = central_jacobian_pert_ndarray_f64(&p, &op, pert);
        let res = vec![
            vec![-4.0, -6.0, 0.0, 0.0, 0.0, 0.0],
            vec![6.0, 5.0, -6.0, 0.0, 0.0, 0.0],
            vec![0.0, 6.0, 5.0, -6.0, 0.0, 0.0],
            vec![0.0, 0.0, 6.0, 5.0, -6.0, 0.0],
            vec![0.0, 0.0, 0.0, 6.0, 5.0, -6.0],
            vec![0.0, 0.0, 0.0, 0.0, 6.0, 9.0],
        ];
        // println!("jacobian:\n{:?}", jacobian);
        // println!("res:\n{:?}", res);
        (0..6)
            .zip(0..6)
            .map(|(i, j)| assert!((res[i][j] - jacobian[(i, j)]).abs() < COMP_ACC))
            .count();
    }

    #[test]
    fn test_central_jacobian_pert_vec_f64_trait() {
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
        let pert = vec![
            PerturbationVector::new()
                .add(0, vec![0, 1])
                .add(3, vec![2, 3, 4]),
            PerturbationVector::new()
                .add(1, vec![0, 1, 2])
                .add(4, vec![3, 4, 5]),
            PerturbationVector::new()
                .add(2, vec![1, 2, 3])
                .add(5, vec![4, 5]),
        ];
        let jacobian = p.central_jacobian_pert(&op, pert);
        let res = vec![
            vec![-4.0, -6.0, 0.0, 0.0, 0.0, 0.0],
            vec![6.0, 5.0, -6.0, 0.0, 0.0, 0.0],
            vec![0.0, 6.0, 5.0, -6.0, 0.0, 0.0],
            vec![0.0, 0.0, 6.0, 5.0, -6.0, 0.0],
            vec![0.0, 0.0, 0.0, 6.0, 5.0, -6.0],
            vec![0.0, 0.0, 0.0, 0.0, 6.0, 9.0],
        ];
        // println!("jacobian:\n{:?}", jacobian);
        // println!("res:\n{:?}", res);
        (0..6)
            .zip(0..6)
            .map(|(i, j)| assert!((res[i][j] - jacobian[i][j]).abs() < COMP_ACC))
            .count();
    }

    #[cfg(feature = "ndarrayl")]
    #[test]
    fn test_central_jacobian_pert_ndarray_f64_trait() {
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
        let pert = vec![
            PerturbationVector::new()
                .add(0, vec![0, 1])
                .add(3, vec![2, 3, 4]),
            PerturbationVector::new()
                .add(1, vec![0, 1, 2])
                .add(4, vec![3, 4, 5]),
            PerturbationVector::new()
                .add(2, vec![1, 2, 3])
                .add(5, vec![4, 5]),
        ];
        let jacobian = p.central_jacobian_pert(&op, pert);
        let res = vec![
            vec![-4.0, -6.0, 0.0, 0.0, 0.0, 0.0],
            vec![6.0, 5.0, -6.0, 0.0, 0.0, 0.0],
            vec![0.0, 6.0, 5.0, -6.0, 0.0, 0.0],
            vec![0.0, 0.0, 6.0, 5.0, -6.0, 0.0],
            vec![0.0, 0.0, 0.0, 6.0, 5.0, -6.0],
            vec![0.0, 0.0, 0.0, 0.0, 6.0, 9.0],
        ];
        // println!("jacobian:\n{:?}", jacobian);
        // println!("res:\n{:?}", res);
        (0..6)
            .zip(0..6)
            .map(|(i, j)| assert!((res[i][j] - jacobian[(i, j)]).abs() < COMP_ACC))
            .count();
    }

    #[test]
    fn test_forward_hessian_vec_f64() {
        let op = |x: &Vec<f64>| x[0] + x[1].powi(2) + x[2] * x[3].powi(2);
        let p = vec![1.0f64, 1.0, 1.0, 1.0];
        let hessian = forward_hessian_vec_f64(&p, &|d| d.forward_diff(&op));
        let res = vec![
            vec![0.0, 0.0, 0.0, 0.0],
            vec![0.0, 2.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 2.0],
            vec![0.0, 0.0, 2.0, 2.0],
        ];
        // println!("hessian:\n{:#?}", hessian);
        // println!("diff:\n{:#?}", diff);
        (0..4)
            .zip(0..4)
            .map(|(i, j)| assert!((res[i][j] - hessian[i][j]).abs() < COMP_ACC))
            .count();
    }

    #[cfg(feature = "ndarrayl")]
    #[test]
    fn test_forward_hessian_ndarray_f64() {
        let op = |x: &ndarray::Array1<f64>| x[0] + x[1].powi(2) + x[2] * x[3].powi(2);
        let p = ndarray::Array1::from_vec(vec![1.0f64, 1.0, 1.0, 1.0]);
        let hessian = forward_hessian_ndarray_f64(&p, &|d| d.forward_diff(&op));
        let res = vec![
            vec![0.0, 0.0, 0.0, 0.0],
            vec![0.0, 2.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 2.0],
            vec![0.0, 0.0, 2.0, 2.0],
        ];
        // println!("hessian:\n{:#?}", hessian);
        // println!("diff:\n{:#?}", diff);
        (0..4)
            .zip(0..4)
            .map(|(i, j)| assert!((res[i][j] - hessian[(i, j)]).abs() < COMP_ACC))
            .count();
    }

    #[test]
    fn test_forward_hessian_vec_f64_trait() {
        let op = |x: &Vec<f64>| x[0] + x[1].powi(2) + x[2] * x[3].powi(2);
        let p = vec![1.0f64, 1.0, 1.0, 1.0];
        let hessian = p.forward_hessian(&|d| d.forward_diff(&op));
        let res = vec![
            vec![0.0, 0.0, 0.0, 0.0],
            vec![0.0, 2.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 2.0],
            vec![0.0, 0.0, 2.0, 2.0],
        ];
        // println!("hessian:\n{:#?}", hessian);
        // println!("diff:\n{:#?}", diff);
        (0..4)
            .zip(0..4)
            .map(|(i, j)| assert!((res[i][j] - hessian[i][j]).abs() < COMP_ACC))
            .count();
    }

    #[cfg(feature = "ndarrayl")]
    #[test]
    fn test_forward_hessian_ndarray_f64_trait() {
        let op = |x: &ndarray::Array1<f64>| x[0] + x[1].powi(2) + x[2] * x[3].powi(2);
        let p = ndarray::Array1::from_vec(vec![1.0f64, 1.0, 1.0, 1.0]);
        let hessian = p.forward_hessian(&|d| d.forward_diff(&op));
        let res = vec![
            vec![0.0, 0.0, 0.0, 0.0],
            vec![0.0, 2.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 2.0],
            vec![0.0, 0.0, 2.0, 2.0],
        ];
        // println!("hessian:\n{:#?}", hessian);
        // println!("diff:\n{:#?}", diff);
        (0..4)
            .zip(0..4)
            .map(|(i, j)| assert!((res[i][j] - hessian[(i, j)]).abs() < COMP_ACC))
            .count();
    }

    #[test]
    fn test_central_hessian_vec_f64() {
        let op = |x: &Vec<f64>| x[0] + x[1].powi(2) + x[2] * x[3].powi(2);
        let p = vec![1.0f64, 1.0, 1.0, 1.0];
        let hessian = central_hessian_vec_f64(&p, &|d| d.central_diff(&op));
        let res = vec![
            vec![0.0, 0.0, 0.0, 0.0],
            vec![0.0, 2.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 2.0],
            vec![0.0, 0.0, 2.0, 2.0],
        ];
        // println!("hessian:\n{:#?}", hessian);
        // println!("diff:\n{:#?}", diff);
        (0..4)
            .zip(0..4)
            .map(|(i, j)| assert!((res[i][j] - hessian[i][j]).abs() < COMP_ACC))
            .count();
    }

    #[cfg(feature = "ndarrayl")]
    #[test]
    fn test_central_hessian_ndarray_f64() {
        let op = |x: &ndarray::Array1<f64>| x[0] + x[1].powi(2) + x[2] * x[3].powi(2);
        let p = ndarray::Array1::from_vec(vec![1.0f64, 1.0, 1.0, 1.0]);
        let hessian = central_hessian_ndarray_f64(&p, &|d| d.central_diff(&op));
        let res = vec![
            vec![0.0, 0.0, 0.0, 0.0],
            vec![0.0, 2.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 2.0],
            vec![0.0, 0.0, 2.0, 2.0],
        ];
        // println!("hessian:\n{:#?}", hessian);
        // println!("diff:\n{:#?}", diff);
        (0..4)
            .zip(0..4)
            .map(|(i, j)| assert!((res[i][j] - hessian[(i, j)]).abs() < COMP_ACC))
            .count();
    }
    //
    // #[test]
    // fn test_central_hessian_vec_f64_trait() {
    //     let op = |x: &Vec<f64>| x[0] + x[1].powi(2) + x[2] * x[3].powi(2);
    //     let p = vec![1.0f64, 1.0, 1.0, 1.0];
    //     let hessian = p.central_hessian(&|d| d.forward_diff(&op));
    //     let res = vec![
    //         vec![0.0, 0.0, 0.0, 0.0],
    //         vec![0.0, 2.0, 0.0, 0.0],
    //         vec![0.0, 0.0, 0.0, 2.0],
    //         vec![0.0, 0.0, 2.0, 2.0],
    //     ];
    //     // println!("hessian:\n{:#?}", hessian);
    //     // println!("diff:\n{:#?}", diff);
    //     (0..4)
    //         .zip(0..4)
    //         .map(|(i, j)| assert!((res[i][j] - hessian[i][j]).abs() < COMP_ACC))
    //         .count();
    // }
    //
    // #[cfg(feature = "ndarrayl")]
    // #[test]
    // fn test_central_hessian_ndarray_f64_trait() {
    //     let op = |x: &ndarray::Array1<f64>| x[0] + x[1].powi(2) + x[2] * x[3].powi(2);
    //     let p = ndarray::Array1::from_vec(vec![1.0f64, 1.0, 1.0, 1.0]);
    //     let hessian = p.central_hessian(&|d| d.forward_diff(&op));
    //     let res = vec![
    //         vec![0.0, 0.0, 0.0, 0.0],
    //         vec![0.0, 2.0, 0.0, 0.0],
    //         vec![0.0, 0.0, 0.0, 2.0],
    //         vec![0.0, 0.0, 2.0, 2.0],
    //     ];
    //     // println!("hessian:\n{:#?}", hessian);
    //     // println!("diff:\n{:#?}", diff);
    //     (0..4)
    //         .zip(0..4)
    //         .map(|(i, j)| assert!((res[i][j] - hessian[(i, j)]).abs() < COMP_ACC))
    //         .count();
    // }
}
