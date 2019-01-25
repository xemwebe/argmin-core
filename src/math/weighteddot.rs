// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::math::ArgminDot;
use crate::math::ArgminWeightedDot;

// /// Dot/scalar product of `T` and `self` weighted by W (p^TWv)
// pub trait ArgminWeightedDot<T, U, V> {
//     /// Dot/scalar product of `T` and `self`
//     fn weighted_dot(&self, w: &V, vec: &T) -> U;
// }

// /// Dot/scalar product of `T` and `self`
// pub trait ArgminDot<T, U> {
//     /// Dot/scalar product of `T` and `self`
//     fn dot(&self, other: &T) -> U;
// }

impl<T, U, V> ArgminWeightedDot<T, U, V> for T
where
    Self: ArgminDot<T, U>,
    V: ArgminDot<T, T>,
{
    #[inline]
    fn weighted_dot(&self, w: &V, v: &T) -> U {
        self.dot(&w.dot(v))
    }
}
