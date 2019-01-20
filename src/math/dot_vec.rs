// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::math::ArgminDot;

macro_rules! make_dot_vec {
    ($t:ty) => {
        impl<'a> ArgminDot<Vec<$t>, $t> for Vec<$t> {
            #[inline]
            fn dot(&self, other: &Vec<$t>) -> $t {
                self.iter().zip(other.iter()).map(|(a, b)| a * b).sum()
            }
        }
    };
}

macro_rules! make_dot_mat {
    ($t:ty) => {
        impl ArgminDot<Vec<$t>, Vec<Vec<$t>>> for Vec<$t> {
            #[inline]
            fn dot(&self, other: &Vec<$t>) -> Vec<Vec<$t>> {
                self.iter()
                    .map(|b| other.iter().map(|a| a * b).collect())
                    .collect()
            }
        }
    };
}

// scalar = <row_vec, col_vec>
make_dot_vec!(f32);
make_dot_vec!(f64);
make_dot_vec!(i8);
make_dot_vec!(i16);
make_dot_vec!(i32);
make_dot_vec!(i64);
make_dot_vec!(u8);
make_dot_vec!(u16);
make_dot_vec!(u32);
make_dot_vec!(u64);
make_dot_vec!(isize);
make_dot_vec!(usize);

// mat = <col_vec, row_vec>
make_dot_mat!(f32);
make_dot_mat!(f64);
make_dot_mat!(i8);
make_dot_mat!(i16);
make_dot_mat!(i32);
make_dot_mat!(i64);
make_dot_mat!(u8);
make_dot_mat!(u16);
make_dot_mat!(u32);
make_dot_mat!(u64);
make_dot_mat!(isize);
make_dot_mat!(usize);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec_vec_i8() {
        let a = vec![1i8, 2, 3];
        let b = vec![4i8, 5, 6];
        let product: i8 = a.dot(&b);
        assert_eq!(product, 32);
    }

    #[test]
    fn test_vec_vec_u8() {
        let a = vec![1u8, 2, 3];
        let b = vec![4u8, 5, 6];
        let product: u8 = a.dot(&b);
        assert_eq!(product, 32);
    }

    #[test]
    fn test_vec_vec_i16() {
        let a = vec![1i16, 2, 3];
        let b = vec![4i16, 5, 6];
        let product: i16 = a.dot(&b);
        assert_eq!(product, 32);
    }

    #[test]
    fn test_vec_vec_u16() {
        let a = vec![1u16, 2, 3];
        let b = vec![4u16, 5, 6];
        let product: u16 = a.dot(&b);
        assert_eq!(product, 32);
    }

    #[test]
    fn test_vec_vec_i32() {
        let a = vec![1i32, 2, 3];
        let b = vec![4i32, 5, 6];
        let product: i32 = a.dot(&b);
        assert_eq!(product, 32);
    }

    #[test]
    fn test_vec_vec_u32() {
        let a = vec![1u32, 2, 3];
        let b = vec![4u32, 5, 6];
        let product: u32 = a.dot(&b);
        assert_eq!(product, 32);
    }

    #[test]
    fn test_vec_vec_i64() {
        let a = vec![1i64, 2, 3];
        let b = vec![4i64, 5, 6];
        let product: i64 = a.dot(&b);
        assert_eq!(product, 32);
    }

    #[test]
    fn test_vec_vec_u64() {
        let a = vec![1u64, 2, 3];
        let b = vec![4u64, 5, 6];
        let product: u64 = a.dot(&b);
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

    #[test]
    fn test_mat_vec_i8() {
        let a = vec![1i8, 2, 3];
        let b = vec![4i8, 5, 6];
        let res = vec![vec![4, 5, 6], vec![8, 10, 12], vec![12, 15, 18]];
        let product: Vec<Vec<i8>> = a.dot(&b);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(product[i][j], res[i][j]);
            }
        }
    }

    #[test]
    fn test_mat_vec_u8() {
        let a = vec![1u8, 2, 3];
        let b = vec![4u8, 5, 6];
        let res = vec![vec![4, 5, 6], vec![8, 10, 12], vec![12, 15, 18]];
        let product: Vec<Vec<u8>> = a.dot(&b);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(product[i][j], res[i][j]);
            }
        }
    }

    #[test]
    fn test_mat_vec_i16() {
        let a = vec![1i16, 2, 3];
        let b = vec![4i16, 5, 6];
        let res = vec![vec![4, 5, 6], vec![8, 10, 12], vec![12, 15, 18]];
        let product: Vec<Vec<i16>> = a.dot(&b);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(product[i][j], res[i][j]);
            }
        }
    }

    #[test]
    fn test_mat_vec_u16() {
        let a = vec![1u16, 2, 3];
        let b = vec![4u16, 5, 6];
        let res = vec![vec![4, 5, 6], vec![8, 10, 12], vec![12, 15, 18]];
        let product: Vec<Vec<u16>> = a.dot(&b);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(product[i][j], res[i][j]);
            }
        }
    }

    #[test]
    fn test_mat_vec_i32() {
        let a = vec![1i32, 2, 3];
        let b = vec![4i32, 5, 6];
        let res = vec![vec![4, 5, 6], vec![8, 10, 12], vec![12, 15, 18]];
        let product: Vec<Vec<i32>> = a.dot(&b);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(product[i][j], res[i][j]);
            }
        }
    }

    #[test]
    fn test_mat_vec_u32() {
        let a = vec![1u32, 2, 3];
        let b = vec![4u32, 5, 6];
        let res = vec![vec![4, 5, 6], vec![8, 10, 12], vec![12, 15, 18]];
        let product: Vec<Vec<u32>> = a.dot(&b);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(product[i][j], res[i][j]);
            }
        }
    }

    #[test]
    fn test_mat_vec_i64() {
        let a = vec![1i64, 2, 3];
        let b = vec![4i64, 5, 6];
        let res = vec![vec![4, 5, 6], vec![8, 10, 12], vec![12, 15, 18]];
        let product: Vec<Vec<i64>> = a.dot(&b);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(product[i][j], res[i][j]);
            }
        }
    }

    #[test]
    fn test_mat_vec_u64() {
        let a = vec![1u64, 2, 3];
        let b = vec![4u64, 5, 6];
        let res = vec![vec![4, 5, 6], vec![8, 10, 12], vec![12, 15, 18]];
        let product: Vec<Vec<u64>> = a.dot(&b);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(product[i][j], res[i][j]);
            }
        }
    }

    #[test]
    fn test_mat_vec_f32() {
        let a = vec![1f32, 2.0, 3.0];
        let b = vec![4f32, 5.0, 6.0];
        let res = vec![
            vec![4.0, 5.0, 6.0],
            vec![8.0, 10.0, 12.0],
            vec![12.0, 15.0, 18.0],
        ];
        let product: Vec<Vec<f32>> = a.dot(&b);
        for i in 0..3 {
            for j in 0..3 {
                assert!((product[i][j] - res[i][j]) < std::f32::EPSILON);
            }
        }
    }

    #[test]
    fn test_mat_vec_f64() {
        let a = vec![1f64, 2.0, 3.0];
        let b = vec![4f64, 5.0, 6.0];
        let res = vec![
            vec![4.0, 5.0, 6.0],
            vec![8.0, 10.0, 12.0],
            vec![12.0, 15.0, 18.0],
        ];
        let product: Vec<Vec<f64>> = a.dot(&b);
        for i in 0..3 {
            for j in 0..3 {
                assert!((product[i][j] - res[i][j]) < std::f64::EPSILON);
            }
        }
    }
}
