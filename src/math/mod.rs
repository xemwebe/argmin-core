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

mod add;
#[cfg(feature = "ndarrayl")]
mod add_ndarray;
mod add_vec;
#[cfg(feature = "ndarrayl")]
mod dot_ndarray;
mod dot_vec;
#[cfg(feature = "ndarrayl")]
mod eye_ndarray;
mod eye_vec;
mod scale;
mod sub;
#[cfg(feature = "ndarrayl")]
mod sub_ndarray;
mod sub_vec;
mod weighteddot;
mod zero;
#[cfg(feature = "ndarrayl")]
mod zero_ndarray;
mod zero_vec;
pub use crate::math::add::*;
#[cfg(feature = "ndarrayl")]
pub use crate::math::add_ndarray::*;
pub use crate::math::add_vec::*;
#[cfg(feature = "ndarrayl")]
pub use crate::math::dot_ndarray::*;
pub use crate::math::dot_vec::*;
#[cfg(feature = "ndarrayl")]
pub use crate::math::eye_ndarray::*;
pub use crate::math::eye_vec::*;
pub use crate::math::scale::*;
pub use crate::math::sub::*;
#[cfg(feature = "ndarrayl")]
pub use crate::math::sub_ndarray::*;
pub use crate::math::sub_vec::*;
pub use crate::math::weighteddot::*;
pub use crate::math::zero::*;
#[cfg(feature = "ndarrayl")]
pub use crate::math::zero_ndarray::*;
pub use crate::math::zero_vec::*;

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

/// Return param vector of all zeros (for now, this is a hack. It should be done better)
pub trait ArgminZero {
    /// Return zero(s)
    fn zero_like(&self) -> Self;
    /// Return zero(s)
    fn zero() -> Self;
}

/// Scale `self` by a `U`
pub trait ArgminScale<U> {
    /// Scale `self` by a `U`
    fn scale(&self, factor: U) -> Self;
}

pub trait ArgminEye {
    fn eye(n: usize) -> Self;
    fn eye_like(&self) -> Self;
}

/// Add a `T` to `self`
pub trait ArgminAdd<T, U> {
    /// Add a `T` to `self`
    fn add(&self, other: &T) -> U;
}

/// Subtract a `T` from `self`
pub trait ArgminSub<T, U> {
    /// Subtract a `T` from `self`
    fn sub(&self, other: &T) -> U;
}

// ---------- REFACTORING MARKER -----------

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

/// Compute the l2-norm (`U`) of `self`
pub trait ArgminNorm<U> {
    /// Compute the l2-norm (`U`) of `self`
    fn norm(&self) -> U;
}

/// Compute the inverse (`T`) of `self`
pub trait ArgminInv<T> {
    fn ainv(&self) -> Result<T, Error>;
}

// Suboptimal: self is moved. ndarray however offers array views...
pub trait ArgminTranspose {
    fn t(self) -> Self;
}

#[cfg(feature = "ndarrayl")]
impl ArgminTranspose for ndarray::Array2<f64> {
    fn t(self) -> Self {
        self.reversed_axes()
    }
}

#[cfg(feature = "ndarrayl")]
impl ArgminTranspose for ndarray::Array2<f32> {
    fn t(self) -> Self {
        self.reversed_axes()
    }
}

// could be more efficient!
impl ArgminTranspose for Vec<Vec<i8>> {
    fn t(self) -> Self {
        let n1 = self.len();
        let n2 = self[0].len();
        let mut out: Vec<Vec<i8>> = vec![vec![0; n2]; n1];
        for i in 0..n1 {
            for j in 0..n2 {
                out[j][i] = self[i][j];
            }
        }
        out
    }
}

// could be more efficient!
impl ArgminTranspose for Vec<Vec<u8>> {
    fn t(self) -> Self {
        let n1 = self.len();
        let n2 = self[0].len();
        let mut out: Vec<Vec<u8>> = vec![vec![0; n2]; n1];
        for i in 0..n1 {
            for j in 0..n2 {
                out[j][i] = self[i][j];
            }
        }
        out
    }
}

// could be more efficient!
impl ArgminTranspose for Vec<Vec<i16>> {
    fn t(self) -> Self {
        let n1 = self.len();
        let n2 = self[0].len();
        let mut out: Vec<Vec<i16>> = vec![vec![0; n2]; n1];
        for i in 0..n1 {
            for j in 0..n2 {
                out[j][i] = self[i][j];
            }
        }
        out
    }
}

// could be more efficient!
impl ArgminTranspose for Vec<Vec<u16>> {
    fn t(self) -> Self {
        let n1 = self.len();
        let n2 = self[0].len();
        let mut out: Vec<Vec<u16>> = vec![vec![0; n2]; n1];
        for i in 0..n1 {
            for j in 0..n2 {
                out[j][i] = self[i][j];
            }
        }
        out
    }
}

// could be more efficient!
impl ArgminTranspose for Vec<Vec<i32>> {
    fn t(self) -> Self {
        let n1 = self.len();
        let n2 = self[0].len();
        let mut out: Vec<Vec<i32>> = vec![vec![0; n2]; n1];
        for i in 0..n1 {
            for j in 0..n2 {
                out[j][i] = self[i][j];
            }
        }
        out
    }
}

// could be more efficient!
impl ArgminTranspose for Vec<Vec<u32>> {
    fn t(self) -> Self {
        let n1 = self.len();
        let n2 = self[0].len();
        let mut out: Vec<Vec<u32>> = vec![vec![0; n2]; n1];
        for i in 0..n1 {
            for j in 0..n2 {
                out[j][i] = self[i][j];
            }
        }
        out
    }
}

// could be more efficient!
impl ArgminTranspose for Vec<Vec<i64>> {
    fn t(self) -> Self {
        let n1 = self.len();
        let n2 = self[0].len();
        let mut out: Vec<Vec<i64>> = vec![vec![0; n2]; n1];
        for i in 0..n1 {
            for j in 0..n2 {
                out[j][i] = self[i][j];
            }
        }
        out
    }
}

// could be more efficient!
impl ArgminTranspose for Vec<Vec<u64>> {
    fn t(self) -> Self {
        let n1 = self.len();
        let n2 = self[0].len();
        let mut out: Vec<Vec<u64>> = vec![vec![0; n2]; n1];
        for i in 0..n1 {
            for j in 0..n2 {
                out[j][i] = self[i][j];
            }
        }
        out
    }
}

// could be more efficient!
impl ArgminTranspose for Vec<Vec<isize>> {
    fn t(self) -> Self {
        let n1 = self.len();
        let n2 = self[0].len();
        let mut out: Vec<Vec<isize>> = vec![vec![0; n2]; n1];
        for i in 0..n1 {
            for j in 0..n2 {
                out[j][i] = self[i][j];
            }
        }
        out
    }
}

// could be more efficient!
impl ArgminTranspose for Vec<Vec<usize>> {
    fn t(self) -> Self {
        let n1 = self.len();
        let n2 = self[0].len();
        let mut out: Vec<Vec<usize>> = vec![vec![0; n2]; n1];
        for i in 0..n1 {
            for j in 0..n2 {
                out[j][i] = self[i][j];
            }
        }
        out
    }
}

// could be more efficient!
impl ArgminTranspose for Vec<Vec<f64>> {
    fn t(self) -> Self {
        let n1 = self.len();
        let n2 = self[0].len();
        let mut out: Vec<Vec<f64>> = vec![vec![0.0; n2]; n1];
        for i in 0..n1 {
            for j in 0..n2 {
                out[j][i] = self[i][j];
            }
        }
        out
    }
}

// could be more efficient!
impl ArgminTranspose for Vec<Vec<f32>> {
    fn t(self) -> Self {
        let n1 = self.len();
        let n2 = self[0].len();
        let mut out: Vec<Vec<f32>> = vec![vec![0.0; n2]; n1];
        for i in 0..n1 {
            for j in 0..n2 {
                out[j][i] = self[i][j];
            }
        }
        out
    }
}

/// Implement a subset of the mathematics traits
macro_rules! make_math {
    ($t:ty, $u:ty, $v:ty) => {
        impl<'a> ArgminScaledAdd<$t, $u> for $v {
            #[inline]
            fn scaled_add(&self, scale: $u, other: &$t) -> $v {
                self.iter()
                    .zip(other.iter())
                    .map(|(a, b)| a + scale * b)
                    .collect()
            }
        }

        impl<'a> ArgminScaledSub<$t, $u> for $v {
            #[inline]
            fn scaled_sub(&self, scale: $u, other: &$t) -> $v {
                self.iter()
                    .zip(other.iter())
                    .map(|(a, b)| a - scale * b)
                    .collect()
            }
        }
    };
}

/// Implement yet another subset of the mathematics traits
macro_rules! make_math3 {
    ($u:ty, $v:ty) => {
        impl<'a> ArgminNorm<$u> for $v {
            #[inline]
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
        impl<'a> ArgminScaledAdd<ndarray::Array1<$t>, $t> for ndarray::Array1<$t> {
            #[inline]
            fn scaled_add(&self, scale: $t, other: &ndarray::Array1<$t>) -> ndarray::Array1<$t> {
                self + &(scale * other)
            }
        }

        impl<'a> ArgminScaledAdd<ndarray::Array2<$t>, $t> for ndarray::Array2<$t> {
            #[inline]
            fn scaled_add(&self, scale: $t, other: &ndarray::Array2<$t>) -> ndarray::Array2<$t> {
                self + &(scale * other)
            }
        }

        impl<'a> ArgminScaledSub<ndarray::Array1<$t>, $t> for ndarray::Array1<$t> {
            #[inline]
            fn scaled_sub(&self, scale: $t, other: &ndarray::Array1<$t>) -> ndarray::Array1<$t> {
                self - &(scale * other)
            }
        }

        impl<'a> ArgminScaledSub<ndarray::Array2<$t>, $t> for ndarray::Array2<$t> {
            #[inline]
            fn scaled_sub(&self, scale: $t, other: &ndarray::Array2<$t>) -> ndarray::Array2<$t> {
                self - &(scale * other)
            }
        }
    };
}

#[cfg(feature = "ndarrayl")]
macro_rules! make_math_ndarray3 {
    ($t:ty) => {
        impl<'a> ArgminNorm<$t> for ndarray::Array1<$t> {
            #[inline]
            fn norm(&self) -> $t {
                self.iter().map(|a| (*a).powi(2)).sum::<$t>().sqrt()
            }
        }

        impl<'a> ArgminInv<ndarray::Array2<$t>> for ndarray::Array2<$t>
        where
            ndarray::Array2<$t>: Inverse,
        {
            #[inline]
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

    #[test]
    fn test_transpose_vec() {
        let mut mat = vec![vec![0.0; 2]; 2];
        let mut t = 1.0f64;
        for i in 0..2 {
            for j in 0..2 {
                mat[i][j] = t;
                t += 1.0;
            }
        }
        let mat2 = mat.clone().t();
        for i in 0..2 {
            for j in 0..2 {
                assert!((mat[i][j] - mat2[j][i]).abs() < std::f64::EPSILON);
            }
        }

        let mut mat = vec![vec![0.0; 2]; 2];
        let mut t = 1.0f32;
        for i in 0..2 {
            for j in 0..2 {
                mat[i][j] = t;
                t += 1.0;
            }
        }
        let mat2 = mat.clone().t();
        for i in 0..2 {
            for j in 0..2 {
                assert!((mat[i][j] - mat2[j][i]).abs() < std::f32::EPSILON);
            }
        }
    }

    #[cfg(feature = "ndarrayl")]
    #[test]
    fn test_transpose_ndarray() {
        let mut mat = ndarray::Array2::zeros((2, 2));
        let mut t = 1.0f64;
        for i in 0..2 {
            for j in 0..2 {
                mat[(i, j)] = t;
                t += 1.0;
            }
        }
        let mat2 = mat.clone().t();
        for i in 0..2 {
            for j in 0..2 {
                assert!((mat[(i, j)] - mat2[(j, i)]).abs() < std::f64::EPSILON);
            }
        }

        let mut mat = ndarray::Array2::zeros((2, 2));
        let mut t = 1.0f32;
        for i in 0..2 {
            for j in 0..2 {
                mat[(i, j)] = t;
                t += 1.0;
            }
        }
        let mat2 = mat.clone().t();
        for i in 0..2 {
            for j in 0..2 {
                assert!((mat[(i, j)] - mat2[(j, i)]).abs() < std::f32::EPSILON);
            }
        }
    }

}
