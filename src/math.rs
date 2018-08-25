// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Math
//!
//! TODO: Documentation.

pub trait ArgminDot<T, U> {
    fn dot(&self, T) -> U;
}

pub trait ArgminAdd<T> {
    fn sub(&self, T) -> Self;
}

pub trait ArgminSub<T> {
    fn sub(&self, T) -> Self;
}

pub trait ArgminScaledAdd<T, U> {
    fn scaled_add(&self, U, T) -> Self;
}

pub trait ArgminScaledSub<T, U> {
    fn scaled_sub(&self, U, T) -> Self;
}

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
