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

/// Add a `T` to `self`
pub trait ArgminAdd<T> {
    /// Add a `T` to `self`
    fn sub(&self, T) -> Self;
}

/// Subtract a `T` from `self`
pub trait ArgminSub<T> {
    /// Subtract a `T` from `self`
    fn sub(&self, T) -> Self;
}

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

/// Implement a subset of the mathematics traits
macro_rules! make_math {
    ($t:ty, $u:ty, $v:ty) => {
        impl<'a> ArgminDot<$t, $u> for $v {
            fn dot(&self, other: $t) -> $u {
                self.iter().zip(other.iter()).map(|(a, b)| a * b).sum()
            }
        }

        impl<'a> ArgminAdd<$t> for $v {
            fn sub(&self, other: $t) -> $v {
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
