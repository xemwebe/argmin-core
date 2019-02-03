// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::math::ArgminInv;
use crate::Error;
use ndarray::Array2;
use ndarray_linalg::Inverse;

macro_rules! make_inv {
    ($t:ty) => {
        impl<'a> ArgminInv<Array2<$t>> for Array2<$t>
        where
            Array2<$t>: Inverse,
        {
            #[inline]
            fn inv(&self) -> Result<Array2<$t>, Error> {
                // Stupid error types...
                Ok(<Self as Inverse>::inv(&self)?)
            }
        }
    };
}

// make_inv!(isize);
// make_inv!(usize);
// make_inv!(i8);
// make_inv!(i16);
// make_inv!(i32);
// make_inv!(i64);
// make_inv!(u8);
// make_inv!(u16);
// make_inv!(u32);
// make_inv!(u64);
make_inv!(f32);
make_inv!(f64);

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_inv_ $t>]() {
                    let a = array![
                        [2 as $t, 5 as $t],
                        [1 as $t, 3 as $t],
                    ];
                    let target = array![
                        [3 as $t, -5 as $t],
                        [-1 as $t, 2 as $t],
                    ];
                    let res = <Array2<$t> as ArgminInv<Array2<$t>>>::inv(&a).unwrap();
                    for i in 0..2 {
                        for j in 0..2 {
                            assert!((((res[(i, j)] - target[(i, j)]) as f64).abs()) < std::f64::EPSILON);
                        }
                    }
                }
            }
        };
    }

    // make_test!(isize);
    // make_test!(usize);
    // make_test!(i8);
    // make_test!(u8);
    // make_test!(i16);
    // make_test!(u16);
    // make_test!(i32);
    // make_test!(u32);
    // make_test!(i64);
    // make_test!(u64);
    make_test!(f32);
    make_test!(f64);
}
