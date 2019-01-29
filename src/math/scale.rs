// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::math::ArgminDot;
use crate::math::ArgminScale;

impl<T, U> ArgminScale<U> for T
where
    T: ArgminDot<U, T>,
{
    #[inline]
    fn scale(&self, factor: U) -> T {
        self.dot(&factor)
    }
}

#[cfg(test)]
mod tests_vec {
    use super::*;
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_vec_ $t>]() {
                    let a = vec![1 as $t, 2 as $t, 3 as $t];
                    let res = a.scale(2 as $t);
                    for i in 0..3 {
                        assert!((((res[i] - (2 as $t) * a[i]) as f64).abs()) < std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_mat_ $t>]() {
                    let a = vec![vec![1 as $t, 2 as $t, 3 as $t], vec![2 as $t, 3 as $t, 4 as $t]];
                    let res = a.scale(2 as $t);
                    for i in 0..3 {
                        for j in 0..2 {
                            assert!((((res[j][i] - (2 as $t) * a[j][i]) as f64).abs()) < std::f64::EPSILON);
                        }
                    }
                }
            }
        };
    }

    make_test!(i8);
    make_test!(u8);
    make_test!(i16);
    make_test!(u16);
    make_test!(i32);
    make_test!(u32);
    make_test!(i64);
    make_test!(u64);
    make_test!(f32);
    make_test!(f64);
}

#[cfg(feature = "ndarrayl")]
#[cfg(test)]
mod tests_ndarray {
    use super::*;
    use ndarray::array;
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_vec_ $t>]() {
                    let a = array![1 as $t, 2 as $t, 3 as $t];
                    let res = a.scale(2 as $t);
                    for i in 0..3 {
                        assert!((((res[i] - (2 as $t) * a[i]) as f64).abs()) < std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_mat_ $t>]() {
                    let a = array![[1 as $t, 2 as $t, 3 as $t], [2 as $t, 3 as $t, 4 as $t]];
                    let res = a.scale(2 as $t);
                    for i in 0..3 {
                        for j in 0..2 {
                            assert!((((res[(j, i)] - (2 as $t) * a[(j, i)]) as f64).abs()) < std::f64::EPSILON);
                        }
                    }
                }
            }
        };
    }

    make_test!(i8);
    make_test!(u8);
    make_test!(i16);
    make_test!(u16);
    make_test!(i32);
    make_test!(u32);
    make_test!(i64);
    make_test!(u64);
    make_test!(f32);
    make_test!(f64);
}