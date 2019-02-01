// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

#[cfg(test)]
mod tests {
    use crate::math::ArgminScaledAdd;
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_scaledadd_vec_ $t>]() {
                    let a = vec![1 as $t, 2 as $t, 3 as $t];
                    let b = 2 as $t;
                    let c = vec![4 as $t, 5 as $t, 6 as $t];
                    let res = a.scaled_add(&b, &c);
                    let target = vec![9 as $t, 12 as $t, 15 as $t];
                    for i in 0..3 {
                        assert!((((res[i] - target[i]) as f64).abs()) < std::f64::EPSILON);
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
