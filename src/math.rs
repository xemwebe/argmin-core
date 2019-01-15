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

use crate::Error;
#[cfg(feature = "ndarrayl")]
use ndarray;
#[cfg(feature = "ndarrayl")]
use ndarray_linalg::Inverse;

/// Modified Cholesky decompositions
pub mod modcholesky {
    //! Modified Cholesky decompositions
    //!
    //! Reexport of `modcholesky` crate.
    pub use modcholesky::*;
}

pub trait ArgminMul<T, U> {
    fn amul(&self, other: &T) -> U;
}

impl ArgminMul<f64, Vec<f64>> for Vec<f64> {
    fn amul(&self, f: &f64) -> Vec<f64> {
        self.iter().map(|x| f * x).collect::<Vec<f64>>()
    }
}

impl<F, T, U> ArgminMul<T, U> for F
where
    F: ArgminDot<T, U>,
{
    fn amul(&self, f: &T) -> U {
        self.dot(f)
    }
}

/// Dot/scalar product of `T` and `self`
pub trait ArgminDot<T, U> {
    /// Dot/scalar product of `T` and `self`
    fn dot(&self, other: &T) -> U;
}

/// Dot/scalar product of `T` and `self` weighted by W (p^TWv)
pub trait ArgminWeightedDot<T, U, V> {
    /// Dot/scalar product of `T` and `self`
    fn weighted_dot(&self, w: &V, vec: &T) -> U;
}

/// TEMPORARY: only for testing!
impl ArgminWeightedDot<Vec<f64>, f64, Vec<Vec<f64>>> for Vec<f64> {
    fn weighted_dot(&self, w: &Vec<Vec<f64>>, v: &Vec<f64>) -> f64 {
        self.dot(&w.iter().map(|x| v.dot(x)).collect::<Vec<f64>>())
    }
}

#[cfg(feature = "ndarrayl")]
impl ArgminWeightedDot<ndarray::Array1<f64>, f64, ndarray::Array2<f64>> for ndarray::Array1<f64> {
    fn weighted_dot(&self, w: &ndarray::Array2<f64>, v: &ndarray::Array1<f64>) -> f64 {
        self.dot(&w.dot(v))
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
    fn add(&self, other: &T) -> Self;
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
    fn sub(&self, other: &T) -> Self;
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
    fn scaled_add(&self, factor: U, vec: &T) -> Self;
}

/// Subtract a `T` scaled by an `U` from `self`
pub trait ArgminScaledSub<T, U> {
    /// Subtract a `T` scaled by an `U` from `self`
    fn scaled_sub(&self, factor: U, vec: &T) -> Self;
}

/// Scale `self` by a `U`
pub trait ArgminScale<U> {
    /// Scale `self` by a `U`
    fn scale(&self, factor: U) -> Self;
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

pub trait ArgminEye {
    fn eye(n: usize) -> Self;
    fn eye_like(&self) -> Self;
}

#[cfg(feature = "ndarrayl")]
impl ArgminEye for ndarray::Array2<f64> {
    fn eye(n: usize) -> Self {
        ndarray::Array2::eye(n)
    }

    fn eye_like(&self) -> Self {
        ndarray::Array2::eye(self.dim().0)
    }
}

#[cfg(feature = "ndarrayl")]
impl ArgminEye for ndarray::Array2<f32> {
    fn eye(n: usize) -> Self {
        ndarray::Array2::eye(n)
    }

    fn eye_like(&self) -> Self {
        ndarray::Array2::eye(self.dim().0)
    }
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

// Hacky: This allows a dot product of the form a*b^T for Vec<Vec<f64>>... Rethink this!
impl ArgminDot<Vec<f64>, Vec<Vec<f64>>> for Vec<f64> {
    fn dot(&self, other: &Vec<f64>) -> Vec<Vec<f64>> {
        other
            .iter()
            .map(|b| self.iter().map(|a| a * b).collect())
            .collect()
    }
}

// Hacky: This allows a dot product of the form a*b^T for Vec<Vec<f32>>... Rethink this!
impl ArgminDot<Vec<f32>, Vec<Vec<f32>>> for Vec<f32> {
    fn dot(&self, other: &Vec<f32>) -> Vec<Vec<f32>> {
        other
            .iter()
            .map(|b| self.iter().map(|a| a * b).collect())
            .collect()
    }
}

// eewwwwww!!! there must be a better way...
#[cfg(feature = "ndarrayl")]
impl ArgminDot<ndarray::Array1<f64>, ndarray::Array2<f64>> for ndarray::Array1<f64> {
    fn dot(&self, other: &ndarray::Array1<f64>) -> ndarray::Array2<f64> {
        let mut out = ndarray::Array2::zeros((self.len(), other.len()));
        for i in 0..self.len() {
            for j in 0..other.len() {
                out[(i, j)] = self[i] * other[j];
            }
        }
        out
    }
}

/// Implement a subset of the mathematics traits
macro_rules! make_math {
    ($t:ty, $u:ty, $v:ty) => {
        impl<'a> ArgminDot<$t, $u> for $v {
            fn dot(&self, other: &$t) -> $u {
                self.iter().zip(other.iter()).map(|(a, b)| a * b).sum()
            }
        }

        impl<'a> ArgminAdd<$t> for $v {
            fn add(&self, other: &$t) -> $v {
                self.iter().zip(other.iter()).map(|(a, b)| a + b).collect()
            }
        }

        impl<'a> ArgminSub<$t> for $v {
            fn sub(&self, other: &$t) -> $v {
                self.iter().zip(other.iter()).map(|(a, b)| a - b).collect()
            }
        }

        impl<'a> ArgminScaledAdd<$t, $u> for $v {
            fn scaled_add(&self, scale: $u, other: &$t) -> $v {
                self.iter()
                    .zip(other.iter())
                    .map(|(a, b)| a + scale * b)
                    .collect()
            }
        }

        impl<'a> ArgminScaledSub<$t, $u> for $v {
            fn scaled_sub(&self, scale: $u, other: &$t) -> $v {
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
            fn dot(&self, other: &ndarray::Array1<$t>) -> $t {
                ndarray::Array1::dot(self, other)
            }
        }

        impl<'a> ArgminDot<ndarray::Array1<$t>, ndarray::Array1<$t>> for ndarray::Array2<$t> {
            fn dot(&self, other: &ndarray::Array1<$t>) -> ndarray::Array1<$t> {
                ndarray::Array2::dot(self, other)
            }
        }

        impl<'a> ArgminDot<ndarray::Array2<$t>, ndarray::Array1<$t>> for ndarray::Array1<$t> {
            fn dot(&self, other: &ndarray::Array2<$t>) -> ndarray::Array1<$t> {
                ndarray::Array1::dot(self, other)
            }
        }

        impl<'a> ArgminDot<ndarray::Array2<$t>, ndarray::Array2<$t>> for ndarray::Array2<$t> {
            fn dot(&self, other: &ndarray::Array2<$t>) -> ndarray::Array2<$t> {
                ndarray::Array2::dot(self, other)
            }
        }

        impl<'a> ArgminAdd<ndarray::Array1<$t>> for ndarray::Array1<$t> {
            fn add(&self, other: &ndarray::Array1<$t>) -> ndarray::Array1<$t> {
                self + other
            }
        }

        impl<'a> ArgminAdd<ndarray::Array2<$t>> for ndarray::Array2<$t> {
            fn add(&self, other: &ndarray::Array2<$t>) -> ndarray::Array2<$t> {
                self + other
            }
        }

        impl<'a> ArgminSub<ndarray::Array1<$t>> for ndarray::Array1<$t> {
            fn sub(&self, other: &ndarray::Array1<$t>) -> ndarray::Array1<$t> {
                self - other
            }
        }

        impl<'a> ArgminSub<ndarray::Array2<$t>> for ndarray::Array2<$t> {
            fn sub(&self, other: &ndarray::Array2<$t>) -> ndarray::Array2<$t> {
                self - other
            }
        }

        impl<'a> ArgminScaledAdd<ndarray::Array1<$t>, $t> for ndarray::Array1<$t> {
            fn scaled_add(&self, scale: $t, other: &ndarray::Array1<$t>) -> ndarray::Array1<$t> {
                self + &(scale * other)
            }
        }

        impl<'a> ArgminScaledAdd<ndarray::Array2<$t>, $t> for ndarray::Array2<$t> {
            fn scaled_add(&self, scale: $t, other: &ndarray::Array2<$t>) -> ndarray::Array2<$t> {
                self + &(scale * other)
            }
        }

        impl<'a> ArgminScaledSub<ndarray::Array1<$t>, $t> for ndarray::Array1<$t> {
            fn scaled_sub(&self, scale: $t, other: &ndarray::Array1<$t>) -> ndarray::Array1<$t> {
                self - &(scale * other)
            }
        }

        impl<'a> ArgminScaledSub<ndarray::Array2<$t>, $t> for ndarray::Array2<$t> {
            fn scaled_sub(&self, scale: $t, other: &ndarray::Array2<$t>) -> ndarray::Array2<$t> {
                self - &(scale * other)
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

    use super::*;
    #[cfg(feature = "ndarrayl")]
    use ndarray::array;

    #[test]
    fn test_dot_vec() {
        let a = vec![1i32, 2, 3];
        let b = vec![4i32, 5, 6];
        let product = a.dot(&b);

        assert_eq!(product, 32);

        let a = vec![1u32, 2, 3];
        let b = vec![4u32, 5, 6];
        let product = a.dot(&b);

        assert_eq!(product, 32);

        let a = vec![1i64, 2, 3];
        let b = vec![4i64, 5, 6];
        let product = a.dot(&b);

        assert_eq!(product, 32);

        let a = vec![1u64, 2, 3];
        let b = vec![4u64, 5, 6];
        let product = a.dot(&b);

        assert_eq!(product, 32);

        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 5.0, 6.0];
        let product = a.dot(&b);

        assert!((product - 32.0).abs() < std::f32::EPSILON);

        let a = vec![1.0f64, 2.0, 3.0];
        let b = vec![4.0f64, 5.0, 6.0];
        let product = a.dot(&b);

        assert!((product - 32.0).abs() < std::f64::EPSILON);
    }

    #[cfg(feature = "ndarrayl")]
    #[test]
    fn test_dot_ndarray() {
        let a = array![1i32, 2, 3];
        let b = array![4i32, 5, 6];
        let product = a.dot(&b);

        assert_eq!(product, 32);

        let a = array![1u32, 2, 3];
        let b = array![4u32, 5, 6];
        let product = a.dot(&b);

        assert_eq!(product, 32);

        let a = array![1i64, 2, 3];
        let b = array![4i64, 5, 6];
        let product = a.dot(&b);

        assert_eq!(product, 32);

        let a = array![1u64, 2, 3];
        let b = array![4u64, 5, 6];
        let product = a.dot(&b);

        assert_eq!(product, 32);

        let a = array![1.0f32, 2.0, 3.0];
        let b = array![4.0f32, 5.0, 6.0];
        let product = a.dot(&b);

        assert!((product - 32.0).abs() < std::f32::EPSILON);

        let a = array![1.0f64, 2.0, 3.0];
        let b = array![4.0f64, 5.0, 6.0];
        let product = a.dot(&b);

        assert!((product - 32.0).abs() < std::f64::EPSILON);
    }

    #[test]
    fn test_amul_vec() {
        let a = vec![1i32, 2, 3];
        let b = vec![4i32, 5, 6];
        let product = a.amul(&b);

        assert_eq!(product, 32);

        let a = vec![1u32, 2, 3];
        let b = vec![4u32, 5, 6];
        let product = a.amul(&b);

        assert_eq!(product, 32);

        let a = vec![1i64, 2, 3];
        let b = vec![4i64, 5, 6];
        let product = a.amul(&b);

        assert_eq!(product, 32);

        let a = vec![1u64, 2, 3];
        let b = vec![4u64, 5, 6];
        let product = a.amul(&b);

        assert_eq!(product, 32);

        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 5.0, 6.0];
        let product = a.amul(&b);

        assert!((product - 32.0).abs() < std::f32::EPSILON);

        let a = vec![1.0f64, 2.0, 3.0];
        let b = vec![4.0f64, 5.0, 6.0];
        let product = a.amul(&b);

        assert!((product - 32.0).abs() < std::f64::EPSILON);
    }

    #[cfg(feature = "ndarrayl")]
    #[test]
    fn test_amul_ndarray() {
        let a = array![1i32, 2, 3];
        let b = array![4i32, 5, 6];
        let product = a.amul(&b);

        assert_eq!(product, 32);

        let a = array![1u32, 2, 3];
        let b = array![4u32, 5, 6];
        let product = a.amul(&b);

        assert_eq!(product, 32);

        let a = array![1i64, 2, 3];
        let b = array![4i64, 5, 6];
        let product = a.amul(&b);

        assert_eq!(product, 32);

        let a = array![1u64, 2, 3];
        let b = array![4u64, 5, 6];
        let product = a.amul(&b);

        assert_eq!(product, 32);

        let a = array![1.0f32, 2.0, 3.0];
        let b = array![4.0f32, 5.0, 6.0];
        let product = a.amul(&b);

        assert!((product - 32.0).abs() < std::f32::EPSILON);

        let a = array![1.0f64, 2.0, 3.0];
        let b = array![4.0f64, 5.0, 6.0];
        let product = a.amul(&b);

        assert!((product - 32.0).abs() < std::f64::EPSILON);
    }
}
