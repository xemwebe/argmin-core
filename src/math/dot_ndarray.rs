// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::math::ArgminDot;
use ndarray::{Array1, Array2};

macro_rules! make_dot_ndarray {
    ($t:ty) => {
        impl ArgminDot<Array1<$t>, $t> for Array1<$t> {
            #[inline]
            fn dot(&self, other: &Array1<$t>) -> $t {
                ndarray::Array1::dot(self, other)
            }
        }

        impl ArgminDot<$t, Array1<$t>> for Array1<$t> {
            #[inline]
            fn dot(&self, other: &$t) -> Array1<$t> {
                self.iter().map(|a| a * other).collect()
            }
        }

        impl<'a> ArgminDot<Array1<$t>, Array1<$t>> for $t {
            #[inline]
            fn dot(&self, other: &Array1<$t>) -> Array1<$t> {
                other.iter().map(|a| a * self).collect()
            }
        }

        impl ArgminDot<Array1<$t>, Array2<$t>> for Array1<$t> {
            #[inline]
            fn dot(&self, other: &Array1<$t>) -> Array2<$t> {
                let mut out = Array2::zeros((self.len(), other.len()));
                for i in 0..self.len() {
                    for j in 0..other.len() {
                        out[(i, j)] = self[i] * other[j];
                    }
                }
                out
            }
        }

        impl ArgminDot<Array1<$t>, Array1<$t>> for Array2<$t> {
            #[inline]
            fn dot(&self, other: &Array1<$t>) -> Array1<$t> {
                ndarray::Array2::dot(self, other)
            }
        }

        impl ArgminDot<Array2<$t>, Array2<$t>> for Array2<$t> {
            #[inline]
            fn dot(&self, other: &Array2<$t>) -> Array2<$t> {
                ndarray::Array2::dot(self, other)
            }
        }
    };
}

make_dot_ndarray!(f32);
make_dot_ndarray!(f64);
make_dot_ndarray!(i8);
make_dot_ndarray!(i16);
make_dot_ndarray!(i32);
make_dot_ndarray!(i64);
make_dot_ndarray!(u8);
make_dot_ndarray!(u16);
make_dot_ndarray!(u32);
make_dot_ndarray!(u64);
make_dot_ndarray!(isize);
make_dot_ndarray!(usize);

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_vec_vec_i8() {
        let a = array![1i8, 2, 3];
        let b = array![4i8, 5, 6];
        let product: i8 = <Array1<i8> as ArgminDot<Array1<i8>, i8>>::dot(&a, &b);
        assert_eq!(product, 32);
    }

    #[test]
    fn test_vec_vec_u8() {
        let a = array![1u8, 2, 3];
        let b = array![4u8, 5, 6];
        let product: u8 = <Array1<u8> as ArgminDot<Array1<u8>, u8>>::dot(&a, &b);
        assert_eq!(product, 32);
    }

    #[test]
    fn test_vec_vec_i16() {
        let a = array![1i16, 2, 3];
        let b = array![4i16, 5, 6];
        let product: i16 = <Array1<i16> as ArgminDot<Array1<i16>, i16>>::dot(&a, &b);
        assert_eq!(product, 32);
    }

    #[test]
    fn test_vec_vec_u16() {
        let a = array![1u16, 2, 3];
        let b = array![4u16, 5, 6];
        let product: u16 = <Array1<u16> as ArgminDot<Array1<u16>, u16>>::dot(&a, &b);
        assert_eq!(product, 32);
    }

    #[test]
    fn test_vec_vec_i32() {
        let a = array![1i32, 2, 3];
        let b = array![4i32, 5, 6];
        let product: i32 = <Array1<i32> as ArgminDot<Array1<i32>, i32>>::dot(&a, &b);
        assert_eq!(product, 32);
    }

    #[test]
    fn test_vec_vec_u32() {
        let a = array![1u32, 2, 3];
        let b = array![4u32, 5, 6];
        let product: u32 = <Array1<u32> as ArgminDot<Array1<u32>, u32>>::dot(&a, &b);
        assert_eq!(product, 32);
    }

    #[test]
    fn test_vec_vec_i64() {
        let a = array![1i64, 2, 3];
        let b = array![4i64, 5, 6];
        let product: i64 = <Array1<i64> as ArgminDot<Array1<i64>, i64>>::dot(&a, &b);
        assert_eq!(product, 32);
    }

    #[test]
    fn test_vec_vec_u64() {
        let a = array![1u64, 2, 3];
        let b = array![4u64, 5, 6];
        let product: u64 = <Array1<u64> as ArgminDot<Array1<u64>, u64>>::dot(&a, &b);
        assert_eq!(product, 32);
    }

    #[test]
    fn test_vec_vec_f32() {
        let a = array![1.0f32, 2.0, 3.0];
        let b = array![4.0f32, 5.0, 6.0];
        let product: f32 = <Array1<f32> as ArgminDot<Array1<f32>, f32>>::dot(&a, &b);
        assert!((product - 32.0f32).abs() < std::f32::EPSILON);
    }

    #[test]
    fn test_vec_vec_f64() {
        let a = array![1.0f64, 2.0, 3.0];
        let b = array![4.0f64, 5.0, 6.0];
        let product: f64 = <Array1<f64> as ArgminDot<Array1<f64>, f64>>::dot(&a, &b);
        assert!((product - 32.0).abs() < std::f64::EPSILON);
    }

    #[test]
    fn test_vec_scalar_i8() {
        let a = array![1i8, 2, 3];
        let b = 2i8;
        let product: Array1<i8> = <Array1<i8> as ArgminDot<i8, Array1<i8>>>::dot(&a, &b);
        let res = array![2i8, 4, 6];
        for i in 0..3 {
            assert_eq!(product[i], res[i]);
        }
    }

    #[test]
    fn test_vec_scalar_u8() {
        let a = array![1u8, 2, 3];
        let b = 2u8;
        let product: Array1<u8> = <Array1<u8> as ArgminDot<u8, Array1<u8>>>::dot(&a, &b);
        let res = array![2u8, 4, 6];
        for i in 0..3 {
            assert_eq!(product[i], res[i]);
        }
    }

    #[test]
    fn test_vec_scalar_i16() {
        let a = array![1i16, 2, 3];
        let b = 2i16;
        let product: Array1<i16> = <Array1<i16> as ArgminDot<i16, Array1<i16>>>::dot(&a, &b);
        let res = array![2i16, 4, 6];
        for i in 0..3 {
            assert_eq!(product[i], res[i]);
        }
    }

    #[test]
    fn test_vec_scalar_u16() {
        let a = array![1u16, 2, 3];
        let b = 2u16;
        let product: Array1<u16> = <Array1<u16> as ArgminDot<u16, Array1<u16>>>::dot(&a, &b);
        let res = array![2u16, 4, 6];
        for i in 0..3 {
            assert_eq!(product[i], res[i]);
        }
    }

    #[test]
    fn test_vec_scalar_i32() {
        let a = array![1i32, 2, 3];
        let b = 2i32;
        let product: Array1<i32> = <Array1<i32> as ArgminDot<i32, Array1<i32>>>::dot(&a, &b);
        let res = array![2i32, 4, 6];
        for i in 0..3 {
            assert_eq!(product[i], res[i]);
        }
    }

    #[test]
    fn test_vec_scalar_u32() {
        let a = array![1u32, 2, 3];
        let b = 2u32;
        let product: Array1<u32> = <Array1<u32> as ArgminDot<u32, Array1<u32>>>::dot(&a, &b);
        let res = array![2u32, 4, 6];
        for i in 0..3 {
            assert_eq!(product[i], res[i]);
        }
    }

    #[test]
    fn test_vec_scalar_i64() {
        let a = array![1i64, 2, 3];
        let b = 2i64;
        let product: Array1<i64> = <Array1<i64> as ArgminDot<i64, Array1<i64>>>::dot(&a, &b);
        let res = array![2i64, 4, 6];
        for i in 0..3 {
            assert_eq!(product[i], res[i]);
        }
    }

    #[test]
    fn test_vec_scalar_u64() {
        let a = array![1u64, 2, 3];
        let b = 2u64;
        let product: Array1<u64> = <Array1<u64> as ArgminDot<u64, Array1<u64>>>::dot(&a, &b);
        let res = array![2u64, 4, 6];
        for i in 0..3 {
            assert_eq!(product[i], res[i]);
        }
    }

    #[test]
    fn test_vec_scalar_f32() {
        let a = array![1f32, 2.0, 3.0];
        let b = 2f32;
        let product: Array1<f32> = <Array1<f32> as ArgminDot<f32, Array1<f32>>>::dot(&a, &b);
        let res = array![2f32, 4.0, 6.0];
        for i in 0..3 {
            assert_eq!(product[i], res[i]);
        }
    }

    #[test]
    fn test_vec_scalar_f64() {
        let a = array![1f64, 2.0, 3.0];
        let b = 2f64;
        let product: Array1<f64> = <Array1<f64> as ArgminDot<f64, Array1<f64>>>::dot(&a, &b);
        let res = array![2f64, 4.0, 6.0];
        for i in 0..3 {
            assert_eq!(product[i], res[i]);
        }
    }

    #[test]
    fn test_scalar_vec_i8() {
        let a = array![1i8, 2, 3];
        let b = 2i8;
        let product: Array1<i8> = <i8 as ArgminDot<Array1<i8>, Array1<i8>>>::dot(&b, &a);
        let res = array![2i8, 4, 6];
        for i in 0..3 {
            assert_eq!(product[i], res[i]);
        }
    }

    #[test]
    fn test_scalar_vec_u8() {
        let a = array![1u8, 2, 3];
        let b = 2u8;
        let product: Array1<u8> = <u8 as ArgminDot<Array1<u8>, Array1<u8>>>::dot(&b, &a);
        let res = array![2u8, 4, 6];
        for i in 0..3 {
            assert_eq!(product[i], res[i]);
        }
    }

    #[test]
    fn test_scalar_vec_i16() {
        let a = array![1i16, 2, 3];
        let b = 2i16;
        let product: Array1<i16> = <i16 as ArgminDot<Array1<i16>, Array1<i16>>>::dot(&b, &a);
        let res = array![2i16, 4, 6];
        for i in 0..3 {
            assert_eq!(product[i], res[i]);
        }
    }

    #[test]
    fn test_scalar_vec_u16() {
        let a = array![1u16, 2, 3];
        let b = 2u16;
        let product: Array1<u16> = <u16 as ArgminDot<Array1<u16>, Array1<u16>>>::dot(&b, &a);
        let res = array![2u16, 4, 6];
        for i in 0..3 {
            assert_eq!(product[i], res[i]);
        }
    }

    #[test]
    fn test_scalar_vec_i32() {
        let a = array![1i32, 2, 3];
        let b = 2i32;
        let product: Array1<i32> = <i32 as ArgminDot<Array1<i32>, Array1<i32>>>::dot(&b, &a);
        let res = array![2i32, 4, 6];
        for i in 0..3 {
            assert_eq!(product[i], res[i]);
        }
    }

    #[test]
    fn test_scalar_vec_u32() {
        let a = array![1u32, 2, 3];
        let b = 2u32;
        let product: Array1<u32> = <u32 as ArgminDot<Array1<u32>, Array1<u32>>>::dot(&b, &a);
        let res = array![2u32, 4, 6];
        for i in 0..3 {
            assert_eq!(product[i], res[i]);
        }
    }

    #[test]
    fn test_scalar_vec_i64() {
        let a = array![1i64, 2, 3];
        let b = 2i64;
        let product: Array1<i64> = <i64 as ArgminDot<Array1<i64>, Array1<i64>>>::dot(&b, &a);
        let res = array![2i64, 4, 6];
        for i in 0..3 {
            assert_eq!(product[i], res[i]);
        }
    }

    #[test]
    fn test_scalar_vec_u64() {
        let a = array![1u64, 2, 3];
        let b = 2u64;
        let product: Array1<u64> = <u64 as ArgminDot<Array1<u64>, Array1<u64>>>::dot(&b, &a);
        let res = array![2u64, 4, 6];
        for i in 0..3 {
            assert_eq!(product[i], res[i]);
        }
    }

    #[test]
    fn test_scalar_vec_f32() {
        let a = array![1f32, 2.0, 3.0];
        let b = 2f32;
        let product: Array1<f32> = <f32 as ArgminDot<Array1<f32>, Array1<f32>>>::dot(&b, &a);
        let res = array![2f32, 4.0, 6.0];
        for i in 0..3 {
            assert_eq!(product[i], res[i]);
        }
    }

    #[test]
    fn test_scalar_vec_f64() {
        let a = array![1f64, 2.0, 3.0];
        let b = 2f64;
        let product: Array1<f64> = <f64 as ArgminDot<Array1<f64>, Array1<f64>>>::dot(&b, &a);
        let res = array![2f64, 4.0, 6.0];
        for i in 0..3 {
            assert_eq!(product[i], res[i]);
        }
    }

    #[test]
    fn test_mat_vec_i8() {
        let a = array![1i8, 2, 3];
        let b = array![4i8, 5, 6];
        let res = vec![vec![4, 5, 6], vec![8, 10, 12], vec![12, 15, 18]];
        let product: Array2<i8> = <Array1<i8> as ArgminDot<Array1<i8>, Array2<i8>>>::dot(&a, &b);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(product[(i, j)], res[i][j]);
            }
        }
    }

    #[test]
    fn test_mat_vec_u8() {
        let a = array![1u8, 2, 3];
        let b = array![4u8, 5, 6];
        let res = vec![vec![4, 5, 6], vec![8, 10, 12], vec![12, 15, 18]];
        let product: Array2<u8> = <Array1<u8> as ArgminDot<Array1<u8>, Array2<u8>>>::dot(&a, &b);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(product[(i, j)], res[i][j]);
            }
        }
    }

    #[test]
    fn test_mat_vec_i16() {
        let a = array![1i16, 2, 3];
        let b = array![4i16, 5, 6];
        let res = vec![vec![4, 5, 6], vec![8, 10, 12], vec![12, 15, 18]];
        let product: Array2<i16> =
            <Array1<i16> as ArgminDot<Array1<i16>, Array2<i16>>>::dot(&a, &b);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(product[(i, j)], res[i][j]);
            }
        }
    }

    #[test]
    fn test_mat_vec_u16() {
        let a = array![1u16, 2, 3];
        let b = array![4u16, 5, 6];
        let res = vec![vec![4, 5, 6], vec![8, 10, 12], vec![12, 15, 18]];
        let product: Array2<u16> =
            <Array1<u16> as ArgminDot<Array1<u16>, Array2<u16>>>::dot(&a, &b);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(product[(i, j)], res[i][j]);
            }
        }
    }

    #[test]
    fn test_mat_vec_i32() {
        let a = array![1i32, 2, 3];
        let b = array![4i32, 5, 6];
        let res = vec![vec![4, 5, 6], vec![8, 10, 12], vec![12, 15, 18]];
        let product: Array2<i32> =
            <Array1<i32> as ArgminDot<Array1<i32>, Array2<i32>>>::dot(&a, &b);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(product[(i, j)], res[i][j]);
            }
        }
    }

    #[test]
    fn test_mat_vec_u32() {
        let a = array![1u32, 2, 3];
        let b = array![4u32, 5, 6];
        let res = vec![vec![4, 5, 6], vec![8, 10, 12], vec![12, 15, 18]];
        let product: Array2<u32> =
            <Array1<u32> as ArgminDot<Array1<u32>, Array2<u32>>>::dot(&a, &b);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(product[(i, j)], res[i][j]);
            }
        }
    }

    #[test]
    fn test_mat_vec_i64() {
        let a = array![1i64, 2, 3];
        let b = array![4i64, 5, 6];
        let res = vec![vec![4, 5, 6], vec![8, 10, 12], vec![12, 15, 18]];
        let product: Array2<i64> =
            <Array1<i64> as ArgminDot<Array1<i64>, Array2<i64>>>::dot(&a, &b);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(product[(i, j)], res[i][j]);
            }
        }
    }

    #[test]
    fn test_mat_vec_u64() {
        let a = array![1u64, 2, 3];
        let b = array![4u64, 5, 6];
        let res = vec![vec![4, 5, 6], vec![8, 10, 12], vec![12, 15, 18]];
        let product: Array2<u64> =
            <Array1<u64> as ArgminDot<Array1<u64>, Array2<u64>>>::dot(&a, &b);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(product[(i, j)], res[i][j]);
            }
        }
    }

    #[test]
    fn test_mat_vec_f32() {
        let a = array![1f32, 2.0, 3.0];
        let b = array![4f32, 5.0, 6.0];
        let res = vec![
            vec![4.0, 5.0, 6.0],
            vec![8.0, 10.0, 12.0],
            vec![12.0, 15.0, 18.0],
        ];
        let product: Array2<f32> =
            <Array1<f32> as ArgminDot<Array1<f32>, Array2<f32>>>::dot(&a, &b);
        for i in 0..3 {
            for j in 0..3 {
                assert!((product[(i, j)] - res[i][j]) < std::f32::EPSILON);
            }
        }
    }

    #[test]
    fn test_mat_vec_f64() {
        let a = array![1f64, 2.0, 3.0];
        let b = array![4f64, 5.0, 6.0];
        let res = vec![
            vec![4.0, 5.0, 6.0],
            vec![8.0, 10.0, 12.0],
            vec![12.0, 15.0, 18.0],
        ];
        let product: Array2<f64> =
            <Array1<f64> as ArgminDot<Array1<f64>, Array2<f64>>>::dot(&a, &b);
        for i in 0..3 {
            for j in 0..3 {
                assert!((product[(i, j)] - res[i][j]) < std::f64::EPSILON);
            }
        }
    }

    #[test]
    fn test_mat_vec_2_i8() {
        let a = array![[1i8, 2, 3], [4, 5, 6], [7, 8, 9]];
        let b = array![1i8, 2, 3];
        let res = vec![14, 32, 50];
        let product: Array1<i8> = <Array2<i8> as ArgminDot<Array1<i8>, Array1<i8>>>::dot(&a, &b);
        for i in 0..3 {
            assert_eq!(product[i], res[i]);
        }
    }

    #[test]
    fn test_mat_vec_2_u8() {
        let a = array![[1u8, 2, 3], [4, 5, 6], [7, 8, 9]];
        let b = array![1u8, 2, 3];
        let res = vec![14, 32, 50];
        let product: Array1<u8> = <Array2<u8> as ArgminDot<Array1<u8>, Array1<u8>>>::dot(&a, &b);
        for i in 0..3 {
            assert_eq!(product[i], res[i]);
        }
    }

    #[test]
    fn test_mat_vec_2_i16() {
        let a = array![[1i16, 2, 3], [4, 5, 6], [7, 8, 9]];
        let b = array![1i16, 2, 3];
        let res = vec![14, 32, 50];
        let product: Array1<i16> =
            <Array2<i16> as ArgminDot<Array1<i16>, Array1<i16>>>::dot(&a, &b);
        for i in 0..3 {
            assert_eq!(product[i], res[i]);
        }
    }

    #[test]
    fn test_mat_vec_2_u16() {
        let a = array![[1u16, 2, 3], [4, 5, 6], [7, 8, 9]];
        let b = array![1u16, 2, 3];
        let res = vec![14, 32, 50];
        let product: Array1<u16> =
            <Array2<u16> as ArgminDot<Array1<u16>, Array1<u16>>>::dot(&a, &b);
        for i in 0..3 {
            assert_eq!(product[i], res[i]);
        }
    }

    #[test]
    fn test_mat_vec_2_i32() {
        let a = array![[1i32, 2, 3], [4, 5, 6], [7, 8, 9]];
        let b = array![1i32, 2, 3];
        let res = vec![14, 32, 50];
        let product: Array1<i32> =
            <Array2<i32> as ArgminDot<Array1<i32>, Array1<i32>>>::dot(&a, &b);
        for i in 0..3 {
            assert_eq!(product[i], res[i]);
        }
    }

    #[test]
    fn test_mat_vec_2_u32() {
        let a = array![[1u32, 2, 3], [4, 5, 6], [7, 8, 9]];
        let b = array![1u32, 2, 3];
        let res = vec![14, 32, 50];
        let product: Array1<u32> =
            <Array2<u32> as ArgminDot<Array1<u32>, Array1<u32>>>::dot(&a, &b);
        for i in 0..3 {
            assert_eq!(product[i], res[i]);
        }
    }

    #[test]
    fn test_mat_vec_2_i64() {
        let a = array![[1i64, 2, 3], [4, 5, 6], [7, 8, 9]];
        let b = array![1i64, 2, 3];
        let res = vec![14, 32, 50];
        let product: Array1<i64> =
            <Array2<i64> as ArgminDot<Array1<i64>, Array1<i64>>>::dot(&a, &b);
        for i in 0..3 {
            assert_eq!(product[i], res[i]);
        }
    }

    #[test]
    fn test_mat_vec_2_u64() {
        let a = array![[1u64, 2, 3], [4, 5, 6], [7, 8, 9]];
        let b = array![1u64, 2, 3];
        let res = vec![14, 32, 50];
        let product: Array1<u64> =
            <Array2<u64> as ArgminDot<Array1<u64>, Array1<u64>>>::dot(&a, &b);
        for i in 0..3 {
            assert_eq!(product[i], res[i]);
        }
    }

    #[test]
    fn test_mat_vec_2_f32() {
        let a = array![[1f32, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let b = array![1f32, 2.0, 3.0];
        let res = vec![14.0, 32.0, 50.0];
        let product: Array1<f32> =
            <Array2<f32> as ArgminDot<Array1<f32>, Array1<f32>>>::dot(&a, &b);
        for i in 0..3 {
            assert!((product[i] - res[i]) < std::f32::EPSILON);
        }
    }

    #[test]
    fn test_mat_vec_2_f64() {
        let a = array![[1f64, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let b = array![1f64, 2.0, 3.0];
        let res = vec![14.0, 32.0, 50.0];
        let product: Array1<f64> =
            <Array2<f64> as ArgminDot<Array1<f64>, Array1<f64>>>::dot(&a, &b);
        for i in 0..3 {
            assert!((product[i] - res[i]) < std::f64::EPSILON);
        }
    }

    #[test]
    fn test_mat_mat_i8() {
        let a = array![[1i8, 2, 3], [4, 5, 6], [3, 2, 1]];
        let b = array![[3i8, 2, 1], [6, 5, 4], [2, 4, 3]];
        let res = array![[21, 24, 18], [54, 57, 42], [23, 20, 14]];
        let product: Array2<i8> = <Array2<i8> as ArgminDot<Array2<i8>, Array2<i8>>>::dot(&a, &b);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(product[(i, j)], res[(i, j)]);
            }
        }
    }

    #[test]
    fn test_mat_mat_u8() {
        let a = array![[1u8, 2, 3], [4, 5, 6], [3, 2, 1]];
        let b = array![[3u8, 2, 1], [6, 5, 4], [2, 4, 3]];
        let res = array![[21, 24, 18], [54, 57, 42], [23, 20, 14]];
        let product: Array2<u8> = <Array2<u8> as ArgminDot<Array2<u8>, Array2<u8>>>::dot(&a, &b);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(product[(i, j)], res[(i, j)]);
            }
        }
    }

    #[test]
    fn test_mat_mat_i16() {
        let a = array![[1i16, 2, 3], [4, 5, 6], [3, 2, 1]];
        let b = array![[3i16, 2, 1], [6, 5, 4], [2, 4, 3]];
        let res = array![[21, 24, 18], [54, 57, 42], [23, 20, 14]];
        let product: Array2<i16> =
            <Array2<i16> as ArgminDot<Array2<i16>, Array2<i16>>>::dot(&a, &b);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(product[(i, j)], res[(i, j)]);
            }
        }
    }

    #[test]
    fn test_mat_mat_u16() {
        let a = array![[1u16, 2, 3], [4, 5, 6], [3, 2, 1]];
        let b = array![[3u16, 2, 1], [6, 5, 4], [2, 4, 3]];
        let res = array![[21, 24, 18], [54, 57, 42], [23, 20, 14]];
        let product: Array2<u16> =
            <Array2<u16> as ArgminDot<Array2<u16>, Array2<u16>>>::dot(&a, &b);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(product[(i, j)], res[(i, j)]);
            }
        }
    }

    #[test]
    fn test_mat_mat_i32() {
        let a = array![[1i32, 2, 3], [4, 5, 6], [3, 2, 1]];
        let b = array![[3i32, 2, 1], [6, 5, 4], [2, 4, 3]];
        let res = array![[21, 24, 18], [54, 57, 42], [23, 20, 14]];
        let product: Array2<i32> =
            <Array2<i32> as ArgminDot<Array2<i32>, Array2<i32>>>::dot(&a, &b);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(product[(i, j)], res[(i, j)]);
            }
        }
    }

    #[test]
    fn test_mat_mat_u32() {
        let a = array![[1u32, 2, 3], [4, 5, 6], [3, 2, 1]];
        let b = array![[3u32, 2, 1], [6, 5, 4], [2, 4, 3]];
        let res = array![[21, 24, 18], [54, 57, 42], [23, 20, 14]];
        let product: Array2<u32> =
            <Array2<u32> as ArgminDot<Array2<u32>, Array2<u32>>>::dot(&a, &b);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(product[(i, j)], res[(i, j)]);
            }
        }
    }

    #[test]
    fn test_mat_mat_i64() {
        let a = array![[1i64, 2, 3], [4, 5, 6], [3, 2, 1]];
        let b = array![[3i64, 2, 1], [6, 5, 4], [2, 4, 3]];
        let res = array![[21, 24, 18], [54, 57, 42], [23, 20, 14]];
        let product: Array2<i64> =
            <Array2<i64> as ArgminDot<Array2<i64>, Array2<i64>>>::dot(&a, &b);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(product[(i, j)], res[(i, j)]);
            }
        }
    }

    #[test]
    fn test_mat_mat_u64() {
        let a = array![[1u64, 2, 3], [4, 5, 6], [3, 2, 1]];
        let b = array![[3u64, 2, 1], [6, 5, 4], [2, 4, 3]];
        let res = array![[21, 24, 18], [54, 57, 42], [23, 20, 14]];
        let product: Array2<u64> =
            <Array2<u64> as ArgminDot<Array2<u64>, Array2<u64>>>::dot(&a, &b);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(product[(i, j)], res[(i, j)]);
            }
        }
    }

    #[test]
    fn test_mat_mat_f32() {
        let a = array![[1f32, 2.0, 3.0], [4.0, 5.0, 6.0], [3.0, 2.0, 1.0]];
        let b = array![[3f32, 2.0, 1.0], [6.0, 5.0, 4.0], [2.0, 4.0, 3.0]];
        let res = array![[21.0, 24.0, 18.0], [54.0, 57.0, 42.0], [23.0, 20.0, 14.0]];
        let product: Array2<f32> =
            <Array2<f32> as ArgminDot<Array2<f32>, Array2<f32>>>::dot(&a, &b);
        for i in 0..3 {
            for j in 0..3 {
                assert!((product[(i, j)] - res[(i, j)]).abs() < std::f32::EPSILON);
            }
        }
    }

    #[test]
    fn test_mat_mat_f64() {
        let a = array![[1f64, 2.0, 3.0], [4.0, 5.0, 6.0], [3.0, 2.0, 1.0]];
        let b = array![[3f64, 2.0, 1.0], [6.0, 5.0, 4.0], [2.0, 4.0, 3.0]];
        let res = array![[21.0, 24.0, 18.0], [54.0, 57.0, 42.0], [23.0, 20.0, 14.0]];
        let product: Array2<f64> =
            <Array2<f64> as ArgminDot<Array2<f64>, Array2<f64>>>::dot(&a, &b);
        for i in 0..3 {
            for j in 0..3 {
                assert!((product[(i, j)] - res[(i, j)]).abs() < std::f64::EPSILON);
            }
        }
    }
}
