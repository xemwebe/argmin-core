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
mod div;
#[cfg(feature = "ndarrayl")]
mod div_ndarray;
mod div_vec;
mod dot;
#[cfg(feature = "ndarrayl")]
mod dot_ndarray;
mod dot_vec;
#[cfg(feature = "ndarrayl")]
mod eye_ndarray;
mod eye_vec;
#[cfg(feature = "ndarrayl")]
mod inv_ndarray;
mod mul;
#[cfg(feature = "ndarrayl")]
mod mul_ndarray;
mod mul_vec;
mod norm;
#[cfg(feature = "ndarrayl")]
mod norm_ndarray;
mod norm_vec;
mod scaledadd;
#[cfg(feature = "ndarrayl")]
mod scaledadd_ndarray;
mod scaledadd_vec;
mod scaledsub;
#[cfg(feature = "ndarrayl")]
mod scaledsub_ndarray;
mod scaledsub_vec;
mod sub;
#[cfg(feature = "ndarrayl")]
mod sub_ndarray;
mod sub_vec;
mod transpose;
#[cfg(feature = "ndarrayl")]
mod transpose_ndarray;
mod transpose_vec;
mod weighteddot;
mod zero;
#[cfg(feature = "ndarrayl")]
mod zero_ndarray;
mod zero_vec;
// #[cfg(feature = "ndarrayl")]
// mod random_ndarray; // TODO
mod random_vec;
// #[cfg(feature = "ndarrayl")]
// mod minmax_ndarray; // TODO
mod minmax_vec;
pub use crate::math::add::*;
#[cfg(feature = "ndarrayl")]
pub use crate::math::add_ndarray::*;
pub use crate::math::add_vec::*;
#[cfg(feature = "ndarrayl")]
pub use crate::math::div_ndarray::*;
pub use crate::math::dot::*;
#[cfg(feature = "ndarrayl")]
pub use crate::math::dot_ndarray::*;
pub use crate::math::dot_vec::*;
#[cfg(feature = "ndarrayl")]
pub use crate::math::eye_ndarray::*;
pub use crate::math::eye_vec::*;
#[cfg(feature = "ndarrayl")]
pub use crate::math::inv_ndarray::*;
pub use crate::math::mul::*;
#[cfg(feature = "ndarrayl")]
pub use crate::math::mul_ndarray::*;
pub use crate::math::mul_vec::*;
pub use crate::math::norm::*;
#[cfg(feature = "ndarrayl")]
pub use crate::math::norm_ndarray::*;
pub use crate::math::norm_vec::*;
pub use crate::math::scaledadd::*;
#[cfg(feature = "ndarrayl")]
pub use crate::math::scaledadd_ndarray::*;
pub use crate::math::scaledadd_vec::*;
pub use crate::math::scaledsub::*;
#[cfg(feature = "ndarrayl")]
pub use crate::math::scaledsub_ndarray::*;
pub use crate::math::scaledsub_vec::*;
pub use crate::math::sub::*;
#[cfg(feature = "ndarrayl")]
pub use crate::math::sub_ndarray::*;
pub use crate::math::sub_vec::*;
pub use crate::math::transpose::*;
#[cfg(feature = "ndarrayl")]
pub use crate::math::transpose_ndarray::*;
pub use crate::math::transpose_vec::*;
pub use crate::math::weighteddot::*;
pub use crate::math::zero::*;
#[cfg(feature = "ndarrayl")]
pub use crate::math::zero_ndarray::*;
pub use crate::math::zero_vec::*;
// #[cfg(feature = "ndarrayl")] // TODO
// pub use crate::math::random_ndarray::*;
pub use crate::math::random_vec::*;
// #[cfg(feature = "ndarrayl")] // TODO
// pub use crate::math::minmax_ndarray::*;
pub use crate::math::minmax_vec::*;

use crate::Error;

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

/// (Pointwise) Multiply a `T` with `self`
pub trait ArgminMul<T, U> {
    /// (Pointwise) Multiply a `T` with `self`
    fn mul(&self, other: &T) -> U;
}

/// (Pointwise) Divide a `T` by `self`
pub trait ArgminDiv<T, U> {
    /// (Pointwise) Divide a `T` by `self`
    fn div(&self, other: &T) -> U;
}

/// Add a `T` scaled by an `U` to `self`
pub trait ArgminScaledAdd<T, U, V> {
    /// Add a `T` scaled by an `U` to `self`
    fn scaled_add(&self, factor: &U, vec: &T) -> V;
}

/// Subtract a `T` scaled by an `U` from `self`
pub trait ArgminScaledSub<T, U, V> {
    /// Subtract a `T` scaled by an `U` from `self`
    fn scaled_sub(&self, factor: &U, vec: &T) -> V;
}

/// Compute the l2-norm (`U`) of `self`
pub trait ArgminNorm<U> {
    /// Compute the l2-norm (`U`) of `self`
    fn norm(&self) -> U;
}

// Suboptimal: self is moved. ndarray however offers array views...
pub trait ArgminTranspose {
    fn t(self) -> Self;
}

/// Compute the inverse (`T`) of `self`
pub trait ArgminInv<T> {
    fn inv(&self) -> Result<T, Error>;
}

pub trait ArgminRandom {
    /// Get a random element between min and max,
    fn rand_from_range(min: &Self, max: &Self) -> Self;
}

pub trait ArgminMinMax {
    /// Select piecewise minimum / maximum
    fn min(x: &Self, y: &Self) -> Self;
    fn max(x: &Self, y: &Self) -> Self;
}