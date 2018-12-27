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

/// Ideally, `EPS_F64` should be set to `EPSILON`; however, this caused numerical  problems which
/// where solved by multiplying it with `4.0`. This may require some investigation.
const EPS_F64: f64 = 4.0 * std::f64::EPSILON;

/// Perturbation Vector for the accelerated computation of the Jacobian.
pub struct PerturbationVector {
    /// x indices
    pub x_idx: Vec<usize>,
    /// correspoding function indices
    pub r_idx: Vec<Vec<usize>>,
}

impl PerturbationVector {
    /// Create a new empty `PerturbationVector`
    pub fn new() -> Self {
        PerturbationVector {
            x_idx: vec![],
            r_idx: vec![],
        }
    }

    /// Add an index `x_idx` and the corresponding function indices `r_idx`
    pub fn add(mut self, x_idx: usize, r_idx: Vec<usize>) -> Self {
        self.x_idx.push(x_idx);
        self.r_idx.push(r_idx);
        self
    }
}

/// A collection of `PerturbationVector`s
pub type PerturbationVectors = Vec<PerturbationVector>;

pub fn forward_diff_vec_f64(p: &Vec<f64>, f: &Fn(&Vec<f64>) -> f64) -> Vec<f64> {
    let fx = (f)(&p);
    let n = p.len();
    (0..n)
        .map(|i| {
            let mut x1 = p.clone();
            x1[i] += EPS_F64.sqrt();
            let fx1 = (f)(&x1);
            (fx1 - fx) / (EPS_F64.sqrt())
        })
        .collect()
}

#[cfg(feature = "ndarrayl")]
pub fn forward_diff_ndarray_f64(
    p: &ndarray::Array1<f64>,
    f: &Fn(&ndarray::Array1<f64>) -> f64,
) -> ndarray::Array1<f64> {
    let fx = (f)(&p);
    let n = p.len();
    (0..n)
        .map(|i| {
            let mut x1 = p.clone();
            x1[i] += EPS_F64.sqrt();
            let fx1 = (f)(&x1);
            (fx1 - fx) / (EPS_F64.sqrt())
        })
        .collect()
}

pub fn forward_jacobian_vec_f64(p: &Vec<f64>, f: &Fn(&Vec<f64>) -> Vec<f64>) -> Vec<Vec<f64>> {
    let fx = (f)(&p);
    let n = p.len();
    (0..n)
        .map(|i| {
            let mut x1 = p.clone();
            x1[i] += EPS_F64.sqrt();
            let fx1 = (f)(&x1);
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
    f: &Fn(&ndarray::Array1<f64>) -> ndarray::Array1<f64>,
) -> ndarray::Array2<f64> {
    let fx = (f)(&p);
    let rn = fx.len();
    let n = p.len();
    let mut out = ndarray::Array2::zeros((rn, n));
    for i in 0..n {
        let mut x1 = p.clone();
        x1[i] += EPS_F64.sqrt();
        let fx1 = (f)(&x1);
        for j in 0..rn {
            out[(j, i)] = (fx1[j] - fx[j]) / EPS_F64.sqrt();
        }
    }
    out
}

pub fn central_diff_vec_f64(p: &Vec<f64>, f: &Fn(&Vec<f64>) -> f64) -> Vec<f64> {
    let n = p.len();
    (0..n)
        .map(|i| {
            let mut x1 = p.clone();
            let mut x2 = p.clone();
            x1[i] += EPS_F64.sqrt();
            x2[i] -= EPS_F64.sqrt();
            let fx1 = (f)(&x1);
            let fx2 = (f)(&x2);
            (fx1 - fx2) / (2.0 * EPS_F64.sqrt())
        })
        .collect()
}

#[cfg(feature = "ndarrayl")]
pub fn central_diff_ndarray_f64(
    p: &ndarray::Array1<f64>,
    f: &Fn(&ndarray::Array1<f64>) -> f64,
) -> ndarray::Array1<f64> {
    let n = p.len();
    (0..n)
        .map(|i| {
            let mut x1 = p.clone();
            let mut x2 = p.clone();
            x1[i] += EPS_F64.sqrt();
            x2[i] -= EPS_F64.sqrt();
            let fx1 = (f)(&x1);
            let fx2 = (f)(&x2);
            (fx1 - fx2) / (2.0 * EPS_F64.sqrt())
        })
        .collect()
}

pub fn forward_jacobian_vec_prod_vec_f64(
    p: &Vec<f64>,
    f: &Fn(&Vec<f64>) -> Vec<f64>,
    x: &Vec<f64>,
) -> Vec<f64> {
    let fx = (f)(&p);
    let x1 = p
        .iter()
        .zip(x.iter())
        .map(|(pi, xi)| pi + EPS_F64.sqrt() * xi)
        .collect();
    let fx1 = (f)(&x1);
    fx1.iter()
        .zip(fx.iter())
        .map(|(a, b)| (a - b) / EPS_F64.sqrt())
        .collect::<Vec<f64>>()
}

#[cfg(feature = "ndarrayl")]
pub fn forward_jacobian_vec_prod_ndarray_f64(
    p: &ndarray::Array1<f64>,
    f: &Fn(&ndarray::Array1<f64>) -> ndarray::Array1<f64>,
    x: &ndarray::Array1<f64>,
) -> ndarray::Array1<f64> {
    let fx = (f)(&p);
    let x1 = p
        .iter()
        .zip(x.iter())
        .map(|(pi, xi)| pi + EPS_F64.sqrt() * xi)
        .collect();
    let fx1 = (f)(&x1);
    (fx1 - fx) / EPS_F64.sqrt()
}

pub fn central_jacobian_vec_f64(p: &Vec<f64>, f: &Fn(&Vec<f64>) -> Vec<f64>) -> Vec<Vec<f64>> {
    let n = p.len();
    (0..n)
        .map(|i| {
            let mut x1 = p.clone();
            let mut x2 = p.clone();
            x1[i] += EPS_F64.sqrt();
            x2[i] -= EPS_F64.sqrt();
            let fx1 = (f)(&x1);
            let fx2 = (f)(&x2);
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
    f: &Fn(&ndarray::Array1<f64>) -> ndarray::Array1<f64>,
) -> ndarray::Array2<f64> {
    let fx = (f)(&p);
    let rn = fx.len();
    let n = p.len();
    let mut out = ndarray::Array2::zeros((rn, n));
    for i in 0..n {
        let mut x1 = p.clone();
        let mut x2 = p.clone();
        x1[i] += EPS_F64.sqrt();
        x2[i] -= EPS_F64.sqrt();
        let fx1 = (f)(&x1);
        let fx2 = (f)(&x2);
        for j in 0..rn {
            out[(j, i)] = (fx1[j] - fx2[j]) / (2.0 * EPS_F64.sqrt());
        }
    }
    out
}

pub fn forward_jacobian_pert_vec_f64(
    p: &Vec<f64>,
    f: &Fn(&Vec<f64>) -> Vec<f64>,
    pert: PerturbationVectors,
) -> Vec<Vec<f64>> {
    let fx = (f)(&p);
    let n = pert.len();
    let mut out = vec![vec![0.0; p.len()]; fx.len()];
    for i in 0..n {
        let mut x1 = p.clone();
        for j in pert[i].x_idx.iter() {
            x1[*j] += EPS_F64.sqrt();
        }
        let fx1 = (f)(&x1);
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
    f: &Fn(&ndarray::Array1<f64>) -> ndarray::Array1<f64>,
    pert: PerturbationVectors,
) -> ndarray::Array2<f64> {
    let fx = (f)(&p);
    let n = pert.len();
    let mut out = ndarray::Array2::zeros((fx.len(), p.len()));
    for i in 0..n {
        let mut x1 = p.clone();
        for j in pert[i].x_idx.iter() {
            x1[*j] += EPS_F64.sqrt();
        }
        let fx1 = (f)(&x1);
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
    f: &Fn(&Vec<f64>) -> Vec<f64>,
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
        let fx1 = (f)(&x1);
        let fx2 = (f)(&x2);
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
    f: &Fn(&ndarray::Array1<f64>) -> ndarray::Array1<f64>,
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
        let fx1 = (f)(&x1);
        let fx2 = (f)(&x2);
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

pub fn forward_hessian_vec_prod_vec_f64(
    p: &Vec<f64>,
    grad: &Fn(&Vec<f64>) -> Vec<f64>,
    x: &Vec<f64>,
) -> Vec<f64> {
    let fx = (grad)(p);
    let out: Vec<f64> = {
        let x1 = p
            .iter()
            .zip(x.iter())
            .map(|(pi, xi)| pi + xi * EPS_F64.sqrt())
            .collect();
        let fx1 = (grad)(&x1);
        fx1.iter()
            .zip(fx.iter())
            .map(|(a, b)| (a - b) / (EPS_F64.sqrt()))
            .collect::<Vec<f64>>()
    };
    out
}

#[cfg(feature = "ndarrayl")]
pub fn forward_hessian_vec_prod_ndarray_f64(
    p: &ndarray::Array1<f64>,
    grad: &Fn(&ndarray::Array1<f64>) -> ndarray::Array1<f64>,
    x: &ndarray::Array1<f64>,
) -> ndarray::Array1<f64> {
    let fx = (grad)(&p);
    let rn = fx.len();
    let mut out = ndarray::Array1::zeros(rn);
    let x1 = p
        .iter()
        .zip(x.iter())
        .map(|(pi, xi)| pi + xi * EPS_F64.sqrt())
        .collect();
    let fx1 = (grad)(&x1);
    for j in 0..rn {
        out[j] = (fx1[j] - fx[j]) / EPS_F64.sqrt();
    }
    out
}

pub fn central_hessian_vec_prod_vec_f64(
    p: &Vec<f64>,
    grad: &Fn(&Vec<f64>) -> Vec<f64>,
    x: &Vec<f64>,
) -> Vec<f64> {
    let out: Vec<f64> = {
        let x1 = p
            .iter()
            .zip(x.iter())
            .map(|(pi, xi)| pi + xi * EPS_F64.sqrt())
            .collect();
        let x2 = p
            .iter()
            .zip(x.iter())
            .map(|(pi, xi)| pi - xi * EPS_F64.sqrt())
            .collect();
        let fx1 = (grad)(&x1);
        let fx2 = (grad)(&x2);
        fx1.iter()
            .zip(fx2.iter())
            .map(|(a, b)| (a - b) / (2.0 * EPS_F64.sqrt()))
            .collect::<Vec<f64>>()
    };
    out
}

#[cfg(feature = "ndarrayl")]
pub fn central_hessian_vec_prod_ndarray_f64(
    p: &ndarray::Array1<f64>,
    grad: &Fn(&ndarray::Array1<f64>) -> ndarray::Array1<f64>,
    x: &ndarray::Array1<f64>,
) -> ndarray::Array1<f64> {
    let rn = p.len();
    let mut out = ndarray::Array1::zeros(rn);
    let x1 = p
        .iter()
        .zip(x.iter())
        .map(|(pi, xi)| pi + xi * EPS_F64.sqrt())
        .collect();
    let x2 = p
        .iter()
        .zip(x.iter())
        .map(|(pi, xi)| pi - xi * EPS_F64.sqrt())
        .collect();
    let fx1 = (grad)(&x1);
    let fx2 = (grad)(&x2);
    for j in 0..rn {
        out[j] = (fx1[j] - fx2[j]) / (2.0 * EPS_F64.sqrt());
    }
    out
}

pub fn forward_hessian_nograd_vec_f64(p: &Vec<f64>, f: &Fn(&Vec<f64>) -> f64) -> Vec<Vec<f64>> {
    let fx = (f)(p);
    let n = p.len();
    let mut out: Vec<Vec<f64>> = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let t = {
                let mut xi = p.clone();
                xi[i] += EPS_F64.sqrt();
                let mut xj = p.clone();
                xj[j] += EPS_F64.sqrt();
                let mut xij = p.clone();
                xij[i] += EPS_F64.sqrt();
                xij[j] += EPS_F64.sqrt();
                let fxi = (f)(&xi);
                let fxj = (f)(&xj);
                let fxij = (f)(&xij);
                (fxij - fxi - fxj + fx) / EPS_F64
            };
            out[i][j] = t;
            out[j][i] = t;
        }
    }
    out
}

#[cfg(feature = "ndarrayl")]
pub fn forward_hessian_nograd_ndarray_f64(
    p: &ndarray::Array1<f64>,
    f: &Fn(&ndarray::Array1<f64>) -> f64,
) -> ndarray::Array2<f64> {
    let fx = (f)(p);
    let n = p.len();
    let mut out = ndarray::Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let t = {
                let mut xi = p.clone();
                xi[i] += EPS_F64.sqrt();
                let mut xj = p.clone();
                xj[j] += EPS_F64.sqrt();
                let mut xij = p.clone();
                xij[i] += EPS_F64.sqrt();
                xij[j] += EPS_F64.sqrt();
                let fxi = (f)(&xi);
                let fxj = (f)(&xj);
                let fxij = (f)(&xij);
                (fxij - fxi - fxj + fx) / EPS_F64
            };
            out[(i, j)] = t;
            out[(j, i)] = t;
        }
    }
    out
}

pub fn forward_hessian_nograd_sparse_vec_f64(
    p: &Vec<f64>,
    f: &Fn(&Vec<f64>) -> f64,
    indices: Vec<(usize, usize)>,
) -> Vec<Vec<f64>> {
    let fx = (f)(p);
    let n = p.len();
    let mut out: Vec<Vec<f64>> = vec![vec![0.0; n]; n];
    for (i, j) in indices {
        let t = {
            let mut xi = p.clone();
            xi[i] += EPS_F64.sqrt();
            let mut xj = p.clone();
            xj[j] += EPS_F64.sqrt();
            let mut xij = p.clone();
            xij[i] += EPS_F64.sqrt();
            xij[j] += EPS_F64.sqrt();
            let fxi = (f)(&xi);
            let fxj = (f)(&xj);
            let fxij = (f)(&xij);
            (fxij - fxi - fxj + fx) / EPS_F64
        };
        out[i][j] = t;
        out[j][i] = t;
    }
    out
}

#[cfg(feature = "ndarrayl")]
pub fn forward_hessian_nograd_sparse_ndarray_f64(
    p: &ndarray::Array1<f64>,
    f: &Fn(&ndarray::Array1<f64>) -> f64,
    indices: Vec<(usize, usize)>,
) -> ndarray::Array2<f64> {
    let fx = (f)(p);
    let n = p.len();
    let mut out = ndarray::Array2::zeros((n, n));
    for (i, j) in indices {
        let t = {
            let mut xi = p.clone();
            xi[i] += EPS_F64.sqrt();
            let mut xj = p.clone();
            xj[j] += EPS_F64.sqrt();
            let mut xij = p.clone();
            xij[i] += EPS_F64.sqrt();
            xij[j] += EPS_F64.sqrt();
            let fxi = (f)(&xi);
            let fxj = (f)(&xj);
            let fxij = (f)(&xij);
            (fxij - fxi - fxj + fx) / EPS_F64
        };
        out[(i, j)] = t;
        out[(j, i)] = t;
    }
    out
}

pub trait ArgminFiniteDiff
where
    Self: Sized,
{
    type Jacobian;
    type Hessian;
    type OperatorOutput;

    /// Forward difference calculated as
    ///
    /// `df/dx_i (x) = (f(x + EPS_F64 * e_i) - f(x))/EPS_F64  \forall i`
    ///
    /// where `e_i` is the `i`th unit vector.
    fn forward_diff(&self, f: &Fn(&Self) -> f64) -> Self;

    /// Central difference calculated as
    ///
    /// `df/dx_i (x) = (f(x + EPS_F64 * e_i) - f(x - EPS_F64 * e_i))/EPS_F64  \forall i`
    ///
    /// where `e_i` is the `i`th unit vector.
    fn central_diff(&self, f: &Fn(&Self) -> f64) -> Self;

    fn forward_jacobian(&self, f: &Fn(&Self) -> Self::OperatorOutput) -> Self::Jacobian;

    fn central_jacobian(&self, f: &Fn(&Self) -> Self::OperatorOutput) -> Self::Jacobian;

    fn forward_jacobian_pert(
        &self,
        f: &Fn(&Self) -> Self::OperatorOutput,
        pert: PerturbationVectors,
    ) -> Self::Jacobian;

    fn central_jacobian_pert(
        &self,
        f: &Fn(&Self) -> Self::OperatorOutput,
        pert: PerturbationVectors,
    ) -> Self::Jacobian;

    fn forward_hessian(&self, grad: &Fn(&Self) -> Self::OperatorOutput) -> Self::Hessian;

    fn central_hessian(&self, grad: &Fn(&Self) -> Self::OperatorOutput) -> Self::Hessian;

    fn forward_hessian_vec_prod(&self, grad: &Fn(&Self) -> Self::OperatorOutput, x: &Self) -> Self;

    fn central_hessian_vec_prod(&self, grad: &Fn(&Self) -> Self::OperatorOutput, x: &Self) -> Self;

    fn forward_hessian_nograd(&self, f: &Fn(&Self) -> f64) -> Self::Hessian;

    fn forward_hessian_nograd_sparse(
        &self,
        f: &Fn(&Self) -> f64,
        indices: Vec<(usize, usize)>,
    ) -> Self::Hessian;
}

impl ArgminFiniteDiff for Vec<f64>
where
    Self: Sized,
{
    type Jacobian = Vec<Vec<f64>>;
    type Hessian = Vec<Vec<f64>>;
    type OperatorOutput = Vec<f64>;

    fn forward_diff(&self, f: &Fn(&Self) -> f64) -> Self {
        forward_diff_vec_f64(self, f)
    }

    fn central_diff(&self, f: &Fn(&Vec<f64>) -> f64) -> Self {
        central_diff_vec_f64(self, f)
    }

    fn forward_jacobian(&self, f: &Fn(&Self) -> Self::OperatorOutput) -> Self::Jacobian {
        forward_jacobian_vec_f64(self, f)
    }

    fn central_jacobian(&self, f: &Fn(&Self) -> Self::OperatorOutput) -> Self::Jacobian {
        central_jacobian_vec_f64(self, f)
    }

    fn forward_jacobian_pert(
        &self,
        f: &Fn(&Self) -> Self::OperatorOutput,
        pert: PerturbationVectors,
    ) -> Self::Jacobian {
        forward_jacobian_pert_vec_f64(self, f, pert)
    }

    fn central_jacobian_pert(
        &self,
        f: &Fn(&Self) -> Self::OperatorOutput,
        pert: PerturbationVectors,
    ) -> Self::Jacobian {
        central_jacobian_pert_vec_f64(self, f, pert)
    }

    fn forward_hessian(&self, grad: &Fn(&Self) -> Self::OperatorOutput) -> Self::Hessian {
        forward_hessian_vec_f64(self, grad)
    }

    fn central_hessian(&self, grad: &Fn(&Self) -> Self::OperatorOutput) -> Self::Hessian {
        central_hessian_vec_f64(self, grad)
    }

    fn forward_hessian_vec_prod(&self, grad: &Fn(&Self) -> Self::OperatorOutput, x: &Self) -> Self {
        forward_hessian_vec_prod_vec_f64(self, grad, x)
    }

    fn central_hessian_vec_prod(&self, grad: &Fn(&Self) -> Self::OperatorOutput, x: &Self) -> Self {
        central_hessian_vec_prod_vec_f64(self, grad, x)
    }

    fn forward_hessian_nograd(&self, f: &Fn(&Self) -> f64) -> Self::Hessian {
        forward_hessian_nograd_vec_f64(self, f)
    }

    fn forward_hessian_nograd_sparse(
        &self,
        f: &Fn(&Self) -> f64,
        indices: Vec<(usize, usize)>,
    ) -> Self::Hessian {
        forward_hessian_nograd_sparse_vec_f64(self, f, indices)
    }
}

#[cfg(feature = "ndarrayl")]
impl ArgminFiniteDiff for ndarray::Array1<f64>
where
    Self: Sized,
{
    type Jacobian = ndarray::Array2<f64>;
    type Hessian = ndarray::Array2<f64>;
    type OperatorOutput = ndarray::Array1<f64>;

    fn forward_diff(&self, f: &Fn(&Self) -> f64) -> Self {
        forward_diff_ndarray_f64(self, f)
    }

    fn central_diff(&self, f: &Fn(&ndarray::Array1<f64>) -> f64) -> Self {
        central_diff_ndarray_f64(self, f)
    }

    fn forward_jacobian(&self, f: &Fn(&Self) -> Self::OperatorOutput) -> Self::Jacobian {
        forward_jacobian_ndarray_f64(self, f)
    }

    fn central_jacobian(&self, f: &Fn(&Self) -> Self::OperatorOutput) -> Self::Jacobian {
        central_jacobian_ndarray_f64(self, f)
    }

    fn forward_jacobian_pert(
        &self,
        f: &Fn(&Self) -> Self::OperatorOutput,
        pert: PerturbationVectors,
    ) -> Self::Jacobian {
        forward_jacobian_pert_ndarray_f64(self, f, pert)
    }

    fn central_jacobian_pert(
        &self,
        f: &Fn(&Self) -> Self::OperatorOutput,
        pert: PerturbationVectors,
    ) -> Self::Jacobian {
        central_jacobian_pert_ndarray_f64(self, f, pert)
    }

    fn forward_hessian(&self, grad: &Fn(&Self) -> Self::OperatorOutput) -> Self::Jacobian {
        forward_hessian_ndarray_f64(self, grad)
    }

    fn central_hessian(&self, grad: &Fn(&Self) -> Self::OperatorOutput) -> Self::Jacobian {
        central_hessian_ndarray_f64(self, grad)
    }

    fn forward_hessian_vec_prod(&self, grad: &Fn(&Self) -> Self::OperatorOutput, x: &Self) -> Self {
        forward_hessian_vec_prod_ndarray_f64(self, grad, x)
    }

    fn central_hessian_vec_prod(&self, grad: &Fn(&Self) -> Self::OperatorOutput, x: &Self) -> Self {
        central_hessian_vec_prod_ndarray_f64(self, grad, x)
    }

    fn forward_hessian_nograd(&self, f: &Fn(&Self) -> f64) -> Self::Hessian {
        forward_hessian_nograd_ndarray_f64(self, f)
    }

    fn forward_hessian_nograd_sparse(
        &self,
        f: &Fn(&Self) -> f64,
        indices: Vec<(usize, usize)>,
    ) -> Self::Hessian {
        forward_hessian_nograd_sparse_ndarray_f64(self, f, indices)
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
        let f = |x: &Vec<f64>| {
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
        let jacobian = forward_jacobian_vec_f64(&p, &f);
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
        let f = |x: &Vec<f64>| {
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
        let jacobian = p.forward_jacobian(&f);
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
        let f = |x: &ndarray::Array1<f64>| {
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
        let jacobian = forward_jacobian_ndarray_f64(&p, &f);
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
        let f = |x: &ndarray::Array1<f64>| {
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
        let jacobian = p.forward_jacobian(&f);
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
        let f = |x: &Vec<f64>| {
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
        let jacobian = central_jacobian_vec_f64(&p, &f);
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
        let f = |x: &ndarray::Array1<f64>| {
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
        let jacobian = central_jacobian_ndarray_f64(&p, &f);
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
        let f = |x: &Vec<f64>| {
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
        let jacobian = p.central_jacobian(&f);
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
        let f = |x: &ndarray::Array1<f64>| {
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
        let jacobian = p.central_jacobian(&f);
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
    fn test_forward_jacobian_vec_prod_vec_f64() {
        let f = |x: &Vec<f64>| {
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
        let x = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let jacobian = forward_jacobian_vec_prod_vec_f64(&p, &f, &x);
        let res = vec![8.0, 22.0, 27.0, 32.0, 37.0, 24.0];
        // println!("{:?}", jacobian);
        // the accuracy for this is pretty bad!!
        (0..6)
            .map(|i| assert!((res[i] - jacobian[i]).abs() < 100.0 * COMP_ACC))
            .count();
    }

    #[cfg(feature = "ndarrayl")]
    #[test]
    fn test_forward_jacobian_vec_prod_ndarray_f64() {
        let f = |x: &ndarray::Array1<f64>| {
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
        let x = ndarray::Array1::from_vec(vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let jacobian = forward_jacobian_vec_prod_ndarray_f64(&p, &f, &x);
        let res = vec![8.0, 22.0, 27.0, 32.0, 37.0, 24.0];
        // println!("{:?}", jacobian);
        // the accuracy for this is pretty bad!!
        (0..6)
            .map(|i| assert!((res[i] - jacobian[i]).abs() < 100.0 * COMP_ACC))
            .count();
    }

    #[test]
    fn test_forward_jacobian_pert_vec_f64() {
        let f = |x: &Vec<f64>| {
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
        let jacobian = forward_jacobian_pert_vec_f64(&p, &f, pert);
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
        let f = |x: &ndarray::Array1<f64>| {
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
        let jacobian = forward_jacobian_pert_ndarray_f64(&p, &f, pert);
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
        let f = |x: &Vec<f64>| {
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
        let jacobian = p.forward_jacobian_pert(&f, pert);
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
        let f = |x: &ndarray::Array1<f64>| {
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
        let jacobian = p.forward_jacobian_pert(&f, pert);
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
        let f = |x: &Vec<f64>| {
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
        let jacobian = central_jacobian_pert_vec_f64(&p, &f, pert);
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
        let f = |x: &ndarray::Array1<f64>| {
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
        let jacobian = central_jacobian_pert_ndarray_f64(&p, &f, pert);
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
        let f = |x: &Vec<f64>| {
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
        let jacobian = p.central_jacobian_pert(&f, pert);
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
        let f = |x: &ndarray::Array1<f64>| {
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
        let jacobian = p.central_jacobian_pert(&f, pert);
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
        let f = |x: &Vec<f64>| x[0] + x[1].powi(2) + x[2] * x[3].powi(2);
        let p = vec![1.0f64, 1.0, 1.0, 1.0];
        let hessian = forward_hessian_vec_f64(&p, &|d| d.forward_diff(&f));
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
        let f = |x: &ndarray::Array1<f64>| x[0] + x[1].powi(2) + x[2] * x[3].powi(2);
        let p = ndarray::Array1::from_vec(vec![1.0f64, 1.0, 1.0, 1.0]);
        let hessian = forward_hessian_ndarray_f64(&p, &|d| d.forward_diff(&f));
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
        let f = |x: &Vec<f64>| x[0] + x[1].powi(2) + x[2] * x[3].powi(2);
        let p = vec![1.0f64, 1.0, 1.0, 1.0];
        let hessian = p.forward_hessian(&|d| d.forward_diff(&f));
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
        let f = |x: &ndarray::Array1<f64>| x[0] + x[1].powi(2) + x[2] * x[3].powi(2);
        let p = ndarray::Array1::from_vec(vec![1.0f64, 1.0, 1.0, 1.0]);
        let hessian = p.forward_hessian(&|d| d.forward_diff(&f));
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
        let f = |x: &Vec<f64>| x[0] + x[1].powi(2) + x[2] * x[3].powi(2);
        let p = vec![1.0f64, 1.0, 1.0, 1.0];
        let hessian = central_hessian_vec_f64(&p, &|d| d.central_diff(&f));
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
        let f = |x: &ndarray::Array1<f64>| x[0] + x[1].powi(2) + x[2] * x[3].powi(2);
        let p = ndarray::Array1::from_vec(vec![1.0f64, 1.0, 1.0, 1.0]);
        let hessian = central_hessian_ndarray_f64(&p, &|d| d.central_diff(&f));
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
    fn test_central_hessian_vec_f64_trait() {
        let f = |x: &Vec<f64>| x[0] + x[1].powi(2) + x[2] * x[3].powi(2);
        let p = vec![1.0f64, 1.0, 1.0, 1.0];
        let hessian = p.central_hessian(&|d| d.central_diff(&f));
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
    fn test_central_hessian_ndarray_f64_trait() {
        let f = |x: &ndarray::Array1<f64>| x[0] + x[1].powi(2) + x[2] * x[3].powi(2);
        let p = ndarray::Array1::from_vec(vec![1.0f64, 1.0, 1.0, 1.0]);
        let hessian = p.central_hessian(&|d| d.central_diff(&f));
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
    fn test_forward_hessian_vec_prod_vec_f64() {
        let f = |x: &Vec<f64>| x[0] + x[1].powi(2) + x[2] * x[3].powi(2);
        let p = vec![1.0f64, 1.0, 1.0, 1.0];
        let x = vec![2.0, 3.0, 4.0, 5.0];
        let hessian = forward_hessian_vec_prod_vec_f64(&p, &|d| d.forward_diff(&f), &x);
        let res = vec![0.0, 6.0, 10.0, 18.0];
        // println!("hessian:\n{:#?}", hessian);
        // println!("diff:\n{:#?}", diff);
        (0..4)
            .map(|i| assert!((res[i] - hessian[i]).abs() < COMP_ACC))
            .count();
    }

    #[cfg(feature = "ndarrayl")]
    #[test]
    fn test_forward_hessian_vec_prod_ndarray_f64() {
        let f = |x: &ndarray::Array1<f64>| x[0] + x[1].powi(2) + x[2] * x[3].powi(2);
        let p = ndarray::Array1::from_vec(vec![1.0f64, 1.0, 1.0, 1.0]);
        let x = ndarray::Array1::from_vec(vec![2.0, 3.0, 4.0, 5.0]);
        let hessian = forward_hessian_vec_prod_ndarray_f64(&p, &|d| d.forward_diff(&f), &x);
        let res = vec![0.0, 6.0, 10.0, 18.0];
        // println!("hessian:\n{:#?}", hessian);
        // println!("diff:\n{:#?}", diff);
        (0..4)
            .map(|i| assert!((res[i] - hessian[i]).abs() < COMP_ACC))
            .count();
    }

    #[test]
    fn test_forward_hessian_vec_prod_vec_f64_trait() {
        let f = |x: &Vec<f64>| x[0] + x[1].powi(2) + x[2] * x[3].powi(2);
        let p = vec![1.0f64, 1.0, 1.0, 1.0];
        let x = vec![2.0, 3.0, 4.0, 5.0];
        let hessian = p.forward_hessian_vec_prod(&|d| d.forward_diff(&f), &x);
        let res = vec![0.0, 6.0, 10.0, 18.0];
        // println!("hessian:\n{:#?}", hessian);
        // println!("diff:\n{:#?}", diff);
        (0..4)
            .map(|i| assert!((res[i] - hessian[i]).abs() < COMP_ACC))
            .count();
    }

    #[cfg(feature = "ndarrayl")]
    #[test]
    fn test_forward_hessian_vec_prod_ndarray_f64_trait() {
        let f = |x: &ndarray::Array1<f64>| x[0] + x[1].powi(2) + x[2] * x[3].powi(2);
        let p = ndarray::Array1::from_vec(vec![1.0f64, 1.0, 1.0, 1.0]);
        let x = ndarray::Array1::from_vec(vec![2.0, 3.0, 4.0, 5.0]);
        let hessian = p.forward_hessian_vec_prod(&|d| d.forward_diff(&f), &x);
        let res = vec![0.0, 6.0, 10.0, 18.0];
        // println!("hessian:\n{:#?}", hessian);
        // println!("diff:\n{:#?}", diff);
        (0..4)
            .map(|i| assert!((res[i] - hessian[i]).abs() < COMP_ACC))
            .count();
    }

    #[test]
    fn test_central_hessian_vec_prod_vec_f64() {
        let f = |x: &Vec<f64>| x[0] + x[1].powi(2) + x[2] * x[3].powi(2);
        let p = vec![1.0f64, 1.0, 1.0, 1.0];
        let x = vec![2.0, 3.0, 4.0, 5.0];
        let hessian = central_hessian_vec_prod_vec_f64(&p, &|d| d.forward_diff(&f), &x);
        let res = vec![0.0, 6.0, 10.0, 18.0];
        // println!("hessian:\n{:#?}", hessian);
        // println!("diff:\n{:#?}", diff);
        (0..4)
            .map(|i| assert!((res[i] - hessian[i]).abs() < COMP_ACC))
            .count();
    }

    #[cfg(feature = "ndarrayl")]
    #[test]
    fn test_central_hessian_vec_prod_ndarray_f64() {
        let f = |x: &ndarray::Array1<f64>| x[0] + x[1].powi(2) + x[2] * x[3].powi(2);
        let p = ndarray::Array1::from_vec(vec![1.0f64, 1.0, 1.0, 1.0]);
        let x = ndarray::Array1::from_vec(vec![2.0, 3.0, 4.0, 5.0]);
        let hessian = central_hessian_vec_prod_ndarray_f64(&p, &|d| d.forward_diff(&f), &x);
        let res = vec![0.0, 6.0, 10.0, 18.0];
        // println!("hessian:\n{:#?}", hessian);
        // println!("diff:\n{:#?}", diff);
        (0..4)
            .map(|i| assert!((res[i] - hessian[i]).abs() < COMP_ACC))
            .count();
    }

    #[test]
    fn test_central_hessian_vec_prod_vec_f64_trait() {
        let f = |x: &Vec<f64>| x[0] + x[1].powi(2) + x[2] * x[3].powi(2);
        let p = vec![1.0f64, 1.0, 1.0, 1.0];
        let x = vec![2.0, 3.0, 4.0, 5.0];
        let hessian = p.central_hessian_vec_prod(&|d| d.central_diff(&f), &x);
        let res = vec![0.0, 6.0, 10.0, 18.0];
        // println!("hessian:\n{:#?}", hessian);
        // println!("diff:\n{:#?}", diff);
        (0..4)
            .map(|i| assert!((res[i] - hessian[i]).abs() < COMP_ACC))
            .count();
    }

    #[cfg(feature = "ndarrayl")]
    #[test]
    fn test_central_hessian_vec_prod_ndarray_f64_trait() {
        let f = |x: &ndarray::Array1<f64>| x[0] + x[1].powi(2) + x[2] * x[3].powi(2);
        let p = ndarray::Array1::from_vec(vec![1.0f64, 1.0, 1.0, 1.0]);
        let x = ndarray::Array1::from_vec(vec![2.0, 3.0, 4.0, 5.0]);
        let hessian = p.central_hessian_vec_prod(&|d| d.central_diff(&f), &x);
        let res = vec![0.0, 6.0, 10.0, 18.0];
        // println!("hessian:\n{:#?}", hessian);
        // println!("diff:\n{:#?}", diff);
        (0..4)
            .map(|i| assert!((res[i] - hessian[i]).abs() < COMP_ACC))
            .count();
    }

    #[test]
    fn test_forward_hessian_nograd_vec_f64() {
        let f = |x: &Vec<f64>| x[0] + x[1].powi(2) + x[2] * x[3].powi(2);
        let p = vec![1.0f64, 1.0, 1.0, 1.0];
        let hessian = forward_hessian_nograd_vec_f64(&p, &f);
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
    fn test_forward_hessian_nograd_ndarray_f64() {
        let f = |x: &ndarray::Array1<f64>| x[0] + x[1].powi(2) + x[2] * x[3].powi(2);
        let p = ndarray::Array1::from_vec(vec![1.0f64, 1.0, 1.0, 1.0]);
        let hessian = forward_hessian_nograd_ndarray_f64(&p, &f);
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
    fn test_forward_hessian_nograd_vec_f64_trait() {
        let f = |x: &Vec<f64>| x[0] + x[1].powi(2) + x[2] * x[3].powi(2);
        let p = vec![1.0f64, 1.0, 1.0, 1.0];
        let hessian = p.forward_hessian_nograd(&f);
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
    fn test_forward_hessian_nograd_ndarray_f64_trait() {
        let f = |x: &ndarray::Array1<f64>| x[0] + x[1].powi(2) + x[2] * x[3].powi(2);
        let p = ndarray::Array1::from_vec(vec![1.0f64, 1.0, 1.0, 1.0]);
        let hessian = p.forward_hessian_nograd(&f);
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
    fn test_forward_hessian_nograd_sparse_vec_f64() {
        let f = |x: &Vec<f64>| x[0] + x[1].powi(2) + x[2] * x[3].powi(2);
        let p = vec![1.0f64, 1.0, 1.0, 1.0];
        let indices = vec![(1, 1), (2, 3), (3, 3)];
        let hessian = forward_hessian_nograd_sparse_vec_f64(&p, &f, indices);
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
    fn test_forward_hessian_nograd_sparse_ndarray_f64() {
        let f = |x: &ndarray::Array1<f64>| x[0] + x[1].powi(2) + x[2] * x[3].powi(2);
        let p = ndarray::Array1::from_vec(vec![1.0f64, 1.0, 1.0, 1.0]);
        let indices = vec![(1, 1), (2, 3), (3, 3)];
        let hessian = forward_hessian_nograd_sparse_ndarray_f64(&p, &f, indices);
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
    fn test_forward_hessian_nograd_sparse_vec_f64_trait() {
        let f = |x: &Vec<f64>| x[0] + x[1].powi(2) + x[2] * x[3].powi(2);
        let p = vec![1.0f64, 1.0, 1.0, 1.0];
        let indices = vec![(1, 1), (2, 3), (3, 3)];
        let hessian = p.forward_hessian_nograd_sparse(&f, indices);
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
    fn test_forward_hessian_nograd_sparse_ndarray_f64_trait() {
        let f = |x: &ndarray::Array1<f64>| x[0] + x[1].powi(2) + x[2] * x[3].powi(2);
        let p = ndarray::Array1::from_vec(vec![1.0f64, 1.0, 1.0, 1.0]);
        let indices = vec![(1, 1), (2, 3), (3, 3)];
        let hessian = p.forward_hessian_nograd_sparse(&f, indices);
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
}
