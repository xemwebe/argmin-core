// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::math::ArgminDot;
// use crate::Error;

// Hacky: This allows a dot product of the form a*b^T for Vec<Vec<f64>>... Rethink this!
impl ArgminDot<Vec<f64>, Vec<Vec<f64>>> for Vec<f64> {
    #[inline]
    fn dot(&self, other: &Vec<f64>) -> Vec<Vec<f64>> {
        other
            .iter()
            .map(|b| self.iter().map(|a| a * b).collect())
            .collect()
    }
}

// Hacky: This allows a dot product of the form a*b^T for Vec<Vec<f32>>... Rethink this!
impl ArgminDot<Vec<f32>, Vec<Vec<f32>>> for Vec<f32> {
    #[inline]
    fn dot(&self, other: &Vec<f32>) -> Vec<Vec<f32>> {
        other
            .iter()
            .map(|b| self.iter().map(|a| a * b).collect())
            .collect()
    }
}

macro_rules! make_dot {
    ($t:ty, $u:ty) => {
        impl<'a> ArgminDot<$t, $u> for $t {
            #[inline]
            fn dot(&self, other: &$t) -> $u {
                self.iter().zip(other.iter()).map(|(a, b)| a * b).sum()
            }
        }
    };
}

// Not sure if all of this makes any sense...
make_dot!(Vec<f32>, f32);
make_dot!(Vec<f64>, f64);
make_dot!(Vec<i8>, i8);
make_dot!(Vec<i16>, i16);
make_dot!(Vec<i32>, i32);
make_dot!(Vec<i64>, i64);
make_dot!(Vec<u8>, u8);
make_dot!(Vec<u16>, u16);
make_dot!(Vec<u32>, u32);
make_dot!(Vec<u64>, u64);
make_dot!(Vec<isize>, isize);
make_dot!(Vec<usize>, usize);
