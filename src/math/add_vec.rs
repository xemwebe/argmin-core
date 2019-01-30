// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::math::ArgminAdd;

macro_rules! make_add {
    ($t:ty) => {
        impl ArgminAdd<$t, Vec<$t>> for Vec<$t> {
            #[inline]
            fn add(&self, other: &$t) -> Vec<$t> {
                self.iter().map(|a| a + other).collect()
            }
        }

        impl ArgminAdd<Vec<$t>, Vec<$t>> for $t {
            #[inline]
            fn add(&self, other: &Vec<$t>) -> Vec<$t> {
                other.iter().map(|a| a + self).collect()
            }
        }

        impl ArgminAdd<Vec<$t>, Vec<$t>> for Vec<$t> {
            #[inline]
            fn add(&self, other: &Vec<$t>) -> Vec<$t> {
                assert_eq!(self.len(), other.len());
                self.iter().zip(other.iter()).map(|(a, b)| a + b).collect()
            }
        }

        impl ArgminAdd<Vec<Vec<$t>>, Vec<Vec<$t>>> for Vec<Vec<$t>> {
            #[inline]
            fn add(&self, other: &Vec<Vec<$t>>) -> Vec<Vec<$t>> {
                let sr = self.len();
                let or = other.len();
                assert!(sr > 0);
                // implicitly, or > 0
                assert_eq!(sr, or);
                let sc = self[0].len();
                self.iter()
                    .zip(other.iter())
                    .map(|(a, b)| {
                        assert_eq!(a.len(), sc);
                        assert_eq!(b.len(), sc);
                        <Vec<$t> as ArgminAdd<Vec<$t>, Vec<$t>>>::add(&a, &b)
                    })
                    .collect()
            }
        }
    };
}

make_add!(isize);
make_add!(usize);
make_add!(i8);
make_add!(i16);
make_add!(i32);
make_add!(i64);
make_add!(u8);
make_add!(u16);
make_add!(u32);
make_add!(u64);
make_add!(f32);
make_add!(f64);

#[cfg(test)]
mod tests {
    use super::*;
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_add_vec_scalar_ $t>]() {
                    let a = vec![1 as $t, 4 as $t, 8 as $t];
                    let b = 34 as $t;
                    let target = vec![35 as $t, 38 as $t, 42 as $t];
                    let res = <Vec<$t> as ArgminAdd<$t, Vec<$t>>>::add(&a, &b);
                    for i in 0..3 {
                        assert!(((target[i] - res[i]) as f64).abs() < std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_add_scalar_vec_ $t>]() {
                    let a = vec![1 as $t, 4 as $t, 8 as $t];
                    let b = 34 as $t;
                    let target = vec![35 as $t, 38 as $t, 42 as $t];
                    let res = <$t as ArgminAdd<Vec<$t>, Vec<$t>>>::add(&b, &a);
                    for i in 0..3 {
                        assert!(((target[i] - res[i]) as f64).abs() < std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                fn [<test_add_vec_vec_ $t>]() {
                    let a = vec![1 as $t, 4 as $t, 8 as $t];
                    let b = vec![41 as $t, 38 as $t, 34 as $t];
                    let target = vec![42 as $t, 42 as $t, 42 as $t];
                    let res = <Vec<$t> as ArgminAdd<Vec<$t>, Vec<$t>>>::add(&a, &b);
                    for i in 0..3 {
                        assert!(((target[i] - res[i]) as f64).abs() < std::f64::EPSILON);
                    }
                }
            }

            item! {
                #[test]
                #[should_panic]
                fn [<test_add_vec_vec_panic_ $t>]() {
                    let a = vec![1 as $t, 4 as $t];
                    let b = vec![41 as $t, 38 as $t, 34 as $t];
                    <Vec<$t> as ArgminAdd<Vec<$t>, Vec<$t>>>::add(&a, &b);
                }
            }
        };
    }

    make_test!(isize);
    make_test!(usize);
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
