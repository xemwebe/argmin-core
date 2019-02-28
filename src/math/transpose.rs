// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::math::ArgminTranspose;

macro_rules! make_transpose {
    ($t:ty) => {
        impl ArgminTranspose for $t {
            #[inline]
            fn t(self) -> $t {
                self
            }
        }
    };
}

make_transpose!(isize);
make_transpose!(usize);
make_transpose!(i8);
make_transpose!(i16);
make_transpose!(i32);
make_transpose!(i64);
make_transpose!(u8);
make_transpose!(u16);
make_transpose!(u32);
make_transpose!(u64);
make_transpose!(f32);
make_transpose!(f64);

#[cfg(test)]
mod tests {
    use super::*;
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_transpose_ $t>]() {
                    let a = 8 as $t;
                    let res = <$t as ArgminTranspose>::t(a);
                    assert!(((8 as $t - res) as f64).abs() < std::f64::EPSILON);
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