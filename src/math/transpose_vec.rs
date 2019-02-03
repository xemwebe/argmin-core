// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

// Note: This is not really the preferred way I think. Maybe this should also be implemented for
// ArrayViews, which would probably make it more efficient.

use crate::math::ArgminTranspose;

macro_rules! make_add {
    ($t:ty) => {
        impl ArgminTranspose for Vec<Vec<$t>> {
            fn t(self) -> Self {
                let n1 = self.len();
                let n2 = self[0].len();
                // let mut out: Vec<Vec<$t>> = vec![vec![0; n1]; n2];
                let mut v = Vec::with_capacity(n1);
                unsafe {
                    v.set_len(n1);
                }
                let mut out = vec![v; n2];
                for i in 0..n1 {
                    for j in 0..n2 {
                        out[j][i] = self[i][j];
                    }
                }
                out
            }
        }
    };
}

make_add!(isize);
make_add!(usize);
make_add!(i8);
make_add!(u8);
make_add!(i16);
make_add!(u16);
make_add!(i32);
make_add!(u32);
make_add!(i64);
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
                fn [<test_transpose_2d_1_ $t>]() {
                    let a = vec![
                        vec![1 as $t, 4 as $t],
                        vec![8 as $t, 7 as $t]
                    ];
                    let target = vec![
                        vec![1 as $t, 8 as $t],
                        vec![4 as $t, 7 as $t]
                    ];
                    let res = a.t();
                    for i in 0..2 {
                        for j in 0..2 {
                            assert!(((target[i][j] - res[i][j]) as f64).abs() < std::f64::EPSILON);
                        }
                    }
                }
            }

            item! {
                #[test]
                fn [<test_transpose_2d_2_ $t>]() {
                    let a = vec![
                        vec![1 as $t, 4 as $t],
                        vec![8 as $t, 7 as $t],
                        vec![3 as $t, 6 as $t]
                    ];
                    let target = vec![
                        vec![1 as $t, 8 as $t, 3 as $t],
                        vec![4 as $t, 7 as $t, 6 as $t]
                    ];
                    let res = a.t();
                    for i in 0..2 {
                        for j in 0..3 {
                            assert!(((target[i][j] - res[i][j]) as f64).abs() < std::f64::EPSILON);
                        }
                    }
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
