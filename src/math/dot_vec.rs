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

macro_rules! make_dot_vec {
    ($t:ty, $u:ty) => {
        impl<'a> ArgminDot<$t, $u> for $t {
            #[inline]
            fn dot(&self, other: &$t) -> $u {
                self.iter().zip(other.iter()).map(|(a, b)| a * b).sum()
            }
        }
    };
}

make_dot_vec!(Vec<f32>, f32);
make_dot_vec!(Vec<f64>, f64);
make_dot_vec!(Vec<i8>, i8);
make_dot_vec!(Vec<i16>, i16);
make_dot_vec!(Vec<i32>, i32);
make_dot_vec!(Vec<i64>, i64);
make_dot_vec!(Vec<u8>, u8);
make_dot_vec!(Vec<u16>, u16);
make_dot_vec!(Vec<u32>, u32);
make_dot_vec!(Vec<u64>, u64);
make_dot_vec!(Vec<isize>, isize);
make_dot_vec!(Vec<usize>, usize);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec_vec_i8() {
        let a = vec![1i8, 2, 3];
        let b = vec![4i8, 5, 6];
        let product = a.dot(&b);
        assert_eq!(product, 32);
    }

    #[test]
    fn test_vec_vec_u8() {
        let a = vec![1u8, 2, 3];
        let b = vec![4u8, 5, 6];
        let product = a.dot(&b);
        assert_eq!(product, 32);
    }

    #[test]
    fn test_vec_vec_i16() {
        let a = vec![1i16, 2, 3];
        let b = vec![4i16, 5, 6];
        let product = a.dot(&b);
        assert_eq!(product, 32);
    }

    #[test]
    fn test_vec_vec_u16() {
        let a = vec![1u16, 2, 3];
        let b = vec![4u16, 5, 6];
        let product = a.dot(&b);
        assert_eq!(product, 32);
    }

    #[test]
    fn test_vec_vec_i32() {
        let a = vec![1i32, 2, 3];
        let b = vec![4i32, 5, 6];
        let product = a.dot(&b);
        assert_eq!(product, 32);
    }

    #[test]
    fn test_vec_vec_u32() {
        let a = vec![1u32, 2, 3];
        let b = vec![4u32, 5, 6];
        let product = a.dot(&b);
        assert_eq!(product, 32);
    }

    #[test]
    fn test_vec_vec_i64() {
        let a = vec![1i64, 2, 3];
        let b = vec![4i64, 5, 6];
        let product = a.dot(&b);
        assert_eq!(product, 32);
    }

    #[test]
    fn test_vec_vec_u64() {
        let a = vec![1u64, 2, 3];
        let b = vec![4u64, 5, 6];
        let product = a.dot(&b);
        assert_eq!(product, 32);
    }

    #[test]
    fn test_vec_vec_f32() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 5.0, 6.0];
        let product: f32 = a.dot(&b);
        assert!((product - 32.0f32).abs() < std::f32::EPSILON);
    }

    #[test]
    fn test_vec_vec_f64() {
        let a = vec![1.0f64, 2.0, 3.0];
        let b = vec![4.0f64, 5.0, 6.0];
        let product: f64 = a.dot(&b);
        assert!((product - 32.0).abs() < std::f64::EPSILON);
    }
}
