// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Math
//!
//! Mathematics related traits which some solvers require. This provides an abstraction over
//! different types of parameter vectors. The idea is, that it does not matter whether you would
//! like to use simple `Vec`s, `ndarray`, `nalgebra` or custom defined types: As long as the traits
//! required by the solver are implemented, you should be fine. In this module several of these
//! traits are defined and implemented. These will be extended as needed. They are also already
//! implemented for basic `Vec`s, and will in the future also be implemented for types defined by
//! `ndarray` and `nalgebra`.

// #[cfg(feature = "ndarrayl")]
// use crate::errors::ArgminError;
#[cfg(feature = "ndarrayl")]
use ndarray;
#[cfg(feature = "ndarrayl")]
use ndarray_linalg::Inverse;
use Error;

pub trait ArgminMul<T, U> {
    fn amul(&self, T) -> U;
}

impl ArgminMul<f64, Vec<f64>> for Vec<f64> {
    fn amul(&self, f: f64) -> Vec<f64> {
        self.iter().map(|x| f * x).collect::<Vec<f64>>()
    }
}

impl<F, T, U> ArgminMul<T, U> for F
where
    F: ArgminDot<T, U>,
{
    fn amul(&self, f: T) -> U {
        self.dot(f)
    }
}

#[cfg(feature = "ndarrayl")]
fn swap_columns<T>(mat: &mut ndarray::Array2<T>, idx1: usize, idx2: usize)
where
    ndarray::OwnedRepr<T>: ndarray::Data,
{
    let s = mat.raw_dim();
    for i in 0..s[0] {
        mat.swap((i, idx1), (i, idx2));
    }
}

#[cfg(feature = "ndarrayl")]
fn swap_rows<T>(mat: &mut ndarray::Array2<T>, idx1: usize, idx2: usize)
where
    ndarray::OwnedRepr<T>: ndarray::Data,
{
    let s = mat.raw_dim();
    for i in 0..s[1] {
        mat.swap((idx1, i), (idx2, i));
    }
}

#[cfg(feature = "ndarrayl")]
fn index_of_largest<'a, T>(c: &ndarray::ArrayView1<T>) -> usize
where
    <ndarray::ViewRepr<&'a T> as ndarray::Data>::Elem:
        std::cmp::PartialOrd + num::traits::Signed + Clone,
{
    let mut max = num::abs(c[0].clone());
    let mut max_idx = 0;
    c.iter()
        .enumerate()
        .map(|(i, ci)| {
            let ci = num::abs(ci.clone());
            if ci > max {
                max = ci;
                max_idx = i
            }
        })
        .count();
    max_idx
}

pub trait ModifiedCholesky
where
    Self: Sized,
{
    fn modified_cholesky(&self) -> Result<(Self, Self), Error>;
}

#[cfg(feature = "ndarrayl")]
impl ModifiedCholesky for ndarray::Array2<f64> {
    /// Algorithm 6.5 in "Numerical Optimization" by Nocedal and Wright
    ///
    /// This can certainly be implemented more memory efficiently
    fn modified_cholesky(&self) -> Result<(ndarray::Array2<f64>, ndarray::Array2<f64>), Error> {
        use ndarray::s;
        debug_assert!(self.is_square());
        let n = self.raw_dim()[0];
        let a_diag = self.diag();

        let diag_max = a_diag.fold(0.0, |acc, x| if x.abs() > acc { x.abs() } else { acc });
        let off_diag_max =
            self.indexed_iter()
                .filter(|((i, j), _)| i != j)
                .fold(
                    0.0,
                    |acc, ((_, _), x)| if x.abs() > acc { x.abs() } else { acc },
                );

        let delta = std::f64::EPSILON * 1.0f64.max(diag_max + off_diag_max);
        let beta = (diag_max
            .max(off_diag_max / ((n as f64).powi(2) - 1.0).sqrt())
            .max(std::f64::EPSILON))
        .sqrt();

        let mut c: ndarray::Array2<f64> = ndarray::Array2::zeros(self.raw_dim());
        c.diag_mut().assign(&a_diag);
        let mut l: ndarray::Array2<f64> = ndarray::Array::zeros((n, n));
        let mut d: ndarray::Array1<f64> = ndarray::Array::zeros(n);

        for j in 0..n {
            let max_idx = index_of_largest(&c.diag().slice(s![j..]));
            swap_rows(&mut c, j, j + max_idx);
            swap_columns(&mut c, j, j + max_idx);
            for s in 0..j {
                l[(j, s)] = c[(j, s)] / d[s];
            }

            // for i in (j + 1)..n {
            for i in j..n {
                c[(i, j)] =
                    self[(i, j)] - (&l.slice(s![j, 0..j]) * &c.slice(s![i, 0..j])).scalar_sum();
            }

            let theta =
                if j < (n - 1) {
                    c.slice(s![(j + 1).., j]).fold(0.0, |acc, x| {
                        if (*x).abs() > acc {
                            (*x).abs()
                        } else {
                            acc
                        }
                    })
                } else {
                    0.0
                };

            d[j] = c[(j, j)].abs().max((theta / beta).powi(2)).max(delta);

            // weirdly enough, this seems to be necessary, even though it is not part of the
            // algorithm in the reference. The reason seems to be that d[j] is not available at the
            // beginning of the loop...
            l[(j, j)] = c[(j, j)] / d[j];

            if j < (n - 1) {
                for i in (j + 1)..n {
                    let c2 = c[(i, j)].powi(2);
                    c[(i, i)] -= c2 / d[j];
                }
            }
        }
        // println!("c: {:?}", c);
        // println!("d: {:?}", d);
        let mut dout = ndarray::Array2::eye(n);
        dout.diag_mut().assign(&d);
        Ok((l, dout))
    }
}

pub trait GershgorinCircles {
    fn gershgorin_circles(&self) -> Result<Vec<(f64, f64)>, Error>;
}

#[cfg(feature = "ndarrayl")]
impl GershgorinCircles for ndarray::Array2<f64> {
    fn gershgorin_circles(&self) -> Result<Vec<(f64, f64)>, Error> {
        debug_assert!(self.is_square());
        use ndarray::s;
        let n = self.raw_dim()[0];
        let mut out: Vec<(f64, f64)> = Vec::with_capacity(n);
        for i in 0..n {
            // TODO: do this with slices instead of loops
            let aii = self[(i, i)];
            let mut ri = 0.0;
            let mut ci = 0.0;
            for j in 0..n {
                if i == j {
                    continue;
                }
                ri += self[(i, j)].abs();
                ci += self[(j, i)].abs();
            }
            out.push((aii, ri.min(ci)));
        }
        Ok(out)
    }
}

/// Dot/scalar product of `T` and `self`
pub trait ArgminDot<T, U> {
    /// Dot/scalar product of `T` and `self`
    fn dot(&self, T) -> U;
}

/// Dot/scalar product of `T` and `self` weighted by W (p^TWv)
pub trait ArgminWeightedDot<T, U, V> {
    /// Dot/scalar product of `T` and `self`
    fn weighted_dot(&self, V, T) -> U;
}

/// TEMPORARY: only for testing!
impl ArgminWeightedDot<Vec<f64>, f64, Vec<Vec<f64>>> for Vec<f64> {
    fn weighted_dot(&self, w: Vec<Vec<f64>>, v: Vec<f64>) -> f64 {
        self.dot(w.iter().map(|x| v.dot(x)).collect::<Vec<f64>>())
    }
}

#[cfg(feature = "ndarrayl")]
impl ArgminWeightedDot<ndarray::Array1<f64>, f64, ndarray::Array2<f64>> for ndarray::Array1<f64> {
    fn weighted_dot(&self, w: ndarray::Array2<f64>, v: ndarray::Array1<f64>) -> f64 {
        self.dot(&w.dot(&v))
    }
}

/// Return param vector of all zeros (for now, this is a hack. It should be done better)
pub trait ArgminZero {
    /// Return param vector of all zeros
    fn zero(&self) -> Self;
}

impl ArgminZero for Vec<f64> {
    fn zero(&self) -> Vec<f64> {
        self.iter().map(|_| 0.0).collect::<Self>()
    }
}

#[cfg(feature = "ndarrayl")]
impl ArgminZero for ndarray::Array1<f64> {
    fn zero(&self) -> ndarray::Array1<f64> {
        self.iter().map(|_| 0.0).collect::<Self>()
    }
}

/// Add a `T` to `self`
pub trait ArgminAdd<T> {
    /// Add a `T` to `self`
    fn add(&self, T) -> Self;
}

// would be great if this worked
// impl<T: std::ops::Add<Output = Self> + Clone> ArgminAdd<T> for T {
//     fn add(&self, other: T) -> T {
//         // is this smart?
//         (*self).clone() + other
//     }
// }

/// Subtract a `T` from `self`
pub trait ArgminSub<T> {
    /// Subtract a `T` from `self`
    fn sub(&self, T) -> Self;
}

// would be great if this worked
// impl<T, U> ArgminSub<T> for U
// where
//     T: std::ops::Sub<Output = T> + Clone,
//     U: std::ops::Sub<Output = T> + Clone,
// {
//     default fn sub(&self, other: T) -> Self {
//         // is this smart?
//         ((*self).clone()) - (other)
//     }
// }

/// Add a `T` scaled by an `U` to `self`
pub trait ArgminScaledAdd<T, U> {
    /// Add a `T` scaled by an `U` to `self`
    fn scaled_add(&self, U, T) -> Self;
}

/// Subtract a `T` scaled by an `U` from `self`
pub trait ArgminScaledSub<T, U> {
    /// Subtract a `T` scaled by an `U` from `self`
    fn scaled_sub(&self, U, T) -> Self;
}

/// Scale `self` by a `U`
pub trait ArgminScale<U> {
    /// Scale `self` by a `U`
    fn scale(&self, U) -> Self;
}

/// Compute the l2-norm (`U`) of `self`
pub trait ArgminNorm<U> {
    /// Compute the l2-norm (`U`) of `self`
    fn norm(&self) -> U;
}

/// Compute the inverse (`T`) of `self`
pub trait ArgminInv<T> {
    fn ainv(&self) -> Result<T, Error>;
}

// impl<'a, T> ArgminInv for T where T: Inverse {}

// impl<'a, T> ArgminInv<T> for T
// where
//     T: Inverse,
// {
//     default fn ainv(&self) -> Result<T, Error> {
//         // Stupid error types...
//         Ok(self.inv()?)
//     }
// }
//
/// Implement a subset of the mathematics traits
macro_rules! make_math {
    ($t:ty, $u:ty, $v:ty) => {
        impl<'a> ArgminDot<$t, $u> for $v {
            fn dot(&self, other: $t) -> $u {
                self.iter().zip(other.iter()).map(|(a, b)| a * b).sum()
            }
        }

        impl<'a> ArgminAdd<$t> for $v {
            fn add(&self, other: $t) -> $v {
                self.iter().zip(other.iter()).map(|(a, b)| a + b).collect()
            }
        }

        impl<'a> ArgminSub<$t> for $v {
            fn sub(&self, other: $t) -> $v {
                self.iter().zip(other.iter()).map(|(a, b)| a - b).collect()
            }
        }

        impl<'a> ArgminScaledAdd<$t, $u> for $v {
            fn scaled_add(&self, scale: $u, other: $t) -> $v {
                self.iter()
                    .zip(other.iter())
                    .map(|(a, b)| a + scale * b)
                    .collect()
            }
        }

        impl<'a> ArgminScaledSub<$t, $u> for $v {
            fn scaled_sub(&self, scale: $u, other: $t) -> $v {
                self.iter()
                    .zip(other.iter())
                    .map(|(a, b)| a - scale * b)
                    .collect()
            }
        }
    };
}

/// Implement another subset of the mathematics traits
macro_rules! make_math2 {
    ($u:ty, $v:ty) => {
        impl<'a> ArgminScale<$u> for $v {
            fn scale(&self, scale: $u) -> $v {
                self.iter().map(|a| scale * a).collect()
            }
        }
    };
}

/// Implement yet another subset of the mathematics traits
macro_rules! make_math3 {
    ($u:ty, $v:ty) => {
        impl<'a> ArgminNorm<$u> for $v {
            fn norm(&self) -> $u {
                self.iter().map(|a| a.powi(2)).sum::<$u>().sqrt()
            }
        }
    };
}

/// Implement a subset of the mathematics traits
#[cfg(feature = "ndarrayl")]
macro_rules! make_math_ndarray {
    ($t:ty) => {
        impl<'a> ArgminDot<ndarray::Array1<$t>, $t> for ndarray::Array1<$t> {
            fn dot(&self, other: ndarray::Array1<$t>) -> $t {
                ndarray::Array1::dot(self, &other)
            }
        }

        impl<'a> ArgminDot<ndarray::Array1<$t>, ndarray::Array1<$t>> for ndarray::Array2<$t> {
            fn dot(&self, other: ndarray::Array1<$t>) -> ndarray::Array1<$t> {
                ndarray::Array2::dot(self, &other)
            }
        }

        impl<'a> ArgminDot<ndarray::Array2<$t>, ndarray::Array1<$t>> for ndarray::Array1<$t> {
            fn dot(&self, other: ndarray::Array2<$t>) -> ndarray::Array1<$t> {
                ndarray::Array1::dot(self, &other)
            }
        }

        impl<'a> ArgminDot<ndarray::Array2<$t>, ndarray::Array2<$t>> for ndarray::Array2<$t> {
            fn dot(&self, other: ndarray::Array2<$t>) -> ndarray::Array2<$t> {
                ndarray::Array2::dot(self, &other)
            }
        }

        impl<'a> ArgminAdd<ndarray::Array1<$t>> for ndarray::Array1<$t> {
            fn add(&self, other: ndarray::Array1<$t>) -> ndarray::Array1<$t> {
                self + &other
            }
        }

        impl<'a> ArgminAdd<ndarray::Array2<$t>> for ndarray::Array2<$t> {
            fn add(&self, other: ndarray::Array2<$t>) -> ndarray::Array2<$t> {
                self + &other
            }
        }

        impl<'a> ArgminSub<ndarray::Array1<$t>> for ndarray::Array1<$t> {
            fn sub(&self, other: ndarray::Array1<$t>) -> ndarray::Array1<$t> {
                self - &other
            }
        }

        impl<'a> ArgminSub<ndarray::Array2<$t>> for ndarray::Array2<$t> {
            fn sub(&self, other: ndarray::Array2<$t>) -> ndarray::Array2<$t> {
                self - &other
            }
        }

        impl<'a> ArgminScaledAdd<ndarray::Array1<$t>, $t> for ndarray::Array1<$t> {
            fn scaled_add(&self, scale: $t, other: ndarray::Array1<$t>) -> ndarray::Array1<$t> {
                self + &(scale * &other)
            }
        }

        impl<'a> ArgminScaledAdd<ndarray::Array2<$t>, $t> for ndarray::Array2<$t> {
            fn scaled_add(&self, scale: $t, other: ndarray::Array2<$t>) -> ndarray::Array2<$t> {
                self + &(scale * &other)
            }
        }

        impl<'a> ArgminScaledSub<ndarray::Array1<$t>, $t> for ndarray::Array1<$t> {
            fn scaled_sub(&self, scale: $t, other: ndarray::Array1<$t>) -> ndarray::Array1<$t> {
                self - &(scale * &other)
            }
        }

        impl<'a> ArgminScaledSub<ndarray::Array2<$t>, $t> for ndarray::Array2<$t> {
            fn scaled_sub(&self, scale: $t, other: ndarray::Array2<$t>) -> ndarray::Array2<$t> {
                self - &(scale * &other)
            }
        }

        impl<'a> ArgminScale<$t> for ndarray::Array1<$t> {
            fn scale(&self, scale: $t) -> ndarray::Array1<$t> {
                scale * self
            }
        }

        impl<'a> ArgminScale<$t> for ndarray::Array2<$t> {
            fn scale(&self, scale: $t) -> ndarray::Array2<$t> {
                scale * self
            }
        }
    };
}

#[cfg(feature = "ndarrayl")]
macro_rules! make_math_ndarray3 {
    ($t:ty) => {
        impl<'a> ArgminNorm<$t> for ndarray::Array1<$t> {
            fn norm(&self) -> $t {
                self.iter().map(|a| (*a).powi(2)).sum::<$t>().sqrt()
            }
        }

        impl<'a> ArgminInv<ndarray::Array2<$t>> for ndarray::Array2<$t>
        where
            ndarray::Array2<$t>: Inverse,
        {
            fn ainv(&self) -> Result<ndarray::Array2<$t>, Error> {
                // Stupid error types...
                Ok(self.inv()?)
            }
        }
    };
}

// Not sure if all of this makes any sense...
make_math!(Vec<f32>, f32, Vec<f32>);
make_math!(Vec<f64>, f64, Vec<f64>);
make_math!(Vec<i8>, i8, Vec<i8>);
make_math!(Vec<i16>, i16, Vec<i16>);
make_math!(Vec<i32>, i32, Vec<i32>);
make_math!(Vec<i64>, i64, Vec<i64>);
make_math!(Vec<u8>, u8, Vec<u8>);
make_math!(Vec<u16>, u16, Vec<u16>);
make_math!(Vec<u32>, u32, Vec<u32>);
make_math!(Vec<u64>, u64, Vec<u64>);
make_math!(Vec<isize>, isize, Vec<isize>);
make_math!(Vec<usize>, usize, Vec<usize>);
make_math!(&'a Vec<f32>, f32, Vec<f32>);
make_math!(&'a Vec<f64>, f64, Vec<f64>);
make_math!(&'a Vec<i8>, i8, Vec<i8>);
make_math!(&'a Vec<i16>, i16, Vec<i16>);
make_math!(&'a Vec<i32>, i32, Vec<i32>);
make_math!(&'a Vec<i64>, i64, Vec<i64>);
make_math!(&'a Vec<u8>, u8, Vec<u8>);
make_math!(&'a Vec<u16>, u16, Vec<u16>);
make_math!(&'a Vec<u32>, u32, Vec<u32>);
make_math!(&'a Vec<u64>, u64, Vec<u64>);
make_math!(&'a Vec<isize>, isize, Vec<isize>);
make_math!(&'a Vec<usize>, usize, Vec<usize>);

make_math2!(f32, Vec<f32>);
make_math2!(f64, Vec<f64>);
make_math2!(i8, Vec<i8>);
make_math2!(i16, Vec<i16>);
make_math2!(i32, Vec<i32>);
make_math2!(i64, Vec<i64>);
make_math2!(u8, Vec<u8>);
make_math2!(u16, Vec<u16>);
make_math2!(u32, Vec<u32>);
make_math2!(u64, Vec<u64>);
make_math2!(isize, Vec<isize>);
make_math2!(usize, Vec<usize>);

make_math3!(f32, Vec<f32>);
make_math3!(f64, Vec<f64>);

#[cfg(feature = "ndarrayl")]
make_math_ndarray!(f32);
#[cfg(feature = "ndarrayl")]
make_math_ndarray!(f64);
#[cfg(feature = "ndarrayl")]
make_math_ndarray!(i8);
#[cfg(feature = "ndarrayl")]
make_math_ndarray!(i16);
#[cfg(feature = "ndarrayl")]
make_math_ndarray!(i32);
#[cfg(feature = "ndarrayl")]
make_math_ndarray!(i64);
#[cfg(feature = "ndarrayl")]
make_math_ndarray!(u8);
#[cfg(feature = "ndarrayl")]
make_math_ndarray!(u16);
#[cfg(feature = "ndarrayl")]
make_math_ndarray!(u32);
#[cfg(feature = "ndarrayl")]
make_math_ndarray!(u64);
#[cfg(feature = "ndarrayl")]
make_math_ndarray3!(f32);
#[cfg(feature = "ndarrayl")]
make_math_ndarray3!(f64);

#[cfg(test)]
mod tests {
    #[cfg(feature = "ndarrayl")]
    #[test]
    fn test_swap_columns() {
        let mut a: ndarray::Array2<i64> = ndarray::arr2(&[[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
        super::swap_columns(&mut a, 1, 2);
        let c: ndarray::Array2<i64> = ndarray::arr2(&[[1, 3, 2], [4, 6, 5], [7, 9, 8]]);
        a.iter()
            .zip(c.iter())
            .map(|(x, y)| assert_eq!(*x, *y))
            .count();
        // this should work, but it doesn't.
        // assert_eq!(b, c);
    }

    #[cfg(feature = "ndarrayl")]
    #[test]
    fn test_swap_rows() {
        let mut a: ndarray::Array2<i64> = ndarray::arr2(&[[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
        super::swap_rows(&mut a, 1, 2);
        let c: ndarray::Array2<i64> = ndarray::arr2(&[[1, 2, 3], [7, 8, 9], [4, 5, 6]]);
        a.iter()
            .zip(c.iter())
            .map(|(x, y)| assert_eq!(*x, *y))
            .count();
        // this should work, but it doesn't.
        // assert_eq!(b, c);
    }

    #[cfg(feature = "ndarrayl")]
    #[test]
    fn test_swap_rows_and_columns() {
        let mut a: ndarray::Array2<i64> = ndarray::arr2(&[[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
        super::swap_rows(&mut a, 1, 2);
        super::swap_columns(&mut a, 1, 2);
        let c: ndarray::Array2<i64> = ndarray::arr2(&[[1, 3, 2], [7, 9, 8], [4, 6, 5]]);
        a.iter()
            .zip(c.iter())
            .map(|(x, y)| assert_eq!(*x, *y))
            .count();
        // this should work, but it doesn't.
        // assert_eq!(b, c);
    }

    #[cfg(feature = "ndarrayl")]
    #[test]
    fn test_biggest_index() {
        use ndarray::s;
        let j = 1;
        let a: ndarray::Array2<i64> =
            ndarray::arr2(&[[1, 2, 3, 0], [4, 2, 6, 0], [7, 8, 3, 0], [3, 4, 2, 8]]);
        let idx = super::index_of_largest(&a.diag().slice(s![j..]));
        assert_eq!(idx + j, 3);
        // this should work, but it doesn't.
        // assert_eq!(b, c);
    }

    #[cfg(feature = "ndarrayl")]
    #[test]
    fn test_modified_cholesky() {
        use super::ModifiedCholesky;
        let a: ndarray::Array2<f64> =
            ndarray::arr2(&[[4.0, 2.0, 1.0], [2.0, 6.0, 3.0], [1.0, 3.0, -0.004]]);
        let (l, d) = a.modified_cholesky().unwrap();
        let f = l.dot(&d).dot(&(l.t()));
        let res: ndarray::Array2<f64> =
            ndarray::arr2(&[[4.0, 2.0, 1.0], [2.0, 6.0, 3.0], [1.0, 3.0, 3.004]]);
        assert!(f.all_close(&res, 2.0 * std::f64::EPSILON));
        // let dsqrt = d.map(|x| x.sqrt());
        // let m = l.dot(&dsqrt);
        // println!("l: {:?}", l);
        // println!("d: {:?}", d);
        // println!("f: {:?}", f);
        // println!("m: {:?}", m);
    }

    #[cfg(feature = "ndarrayl")]
    #[test]
    fn test_gershgorin_circles() {
        use super::GershgorinCircles;
        let a: ndarray::Array2<f64> = ndarray::arr2(&[
            [10.0, -1.0, 0.0, 1.0],
            [0.2, 8.0, 0.2, 0.2],
            [1.0, 1.0, 2.0, 1.0],
            [-1.0, -1.0, -1.0, -11.0],
        ]);
        // without considering the columns as well
        // let b: Vec<(f64, f64)> = vec![(10.0, 2.0), (8.0, 0.6), (2.0, 3.0), (-11.0, 3.0)];
        // with considering the columns
        let b: Vec<(f64, f64)> = vec![(10.0, 2.0), (8.0, 0.6), (2.0, 1.2), (-11.0, 2.2)];
        let res = a.gershgorin_circles().unwrap();
        b.iter()
            .zip(res.iter())
            .map(|((x1, y1), (x2, y2))| {
                assert!((x1 - x2).abs() < 2.0 * std::f64::EPSILON);
                assert!((y1 - y2).abs() < 2.0 * std::f64::EPSILON);
            })
            .count();
    }
}
