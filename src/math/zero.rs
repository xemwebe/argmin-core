// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::math::ArgminZero;
use num::Zero;

macro_rules! make_zero {
    ($t:ty) => {
        impl ArgminZero for $t {
            #[inline]
            fn zero_like(&self) -> $t {
                0 as $t
            }

            #[inline]
            fn zero() -> $t {
                0 as $t
            }
        }
    };
}

make_zero!(f32);
make_zero!(f64);
make_zero!(i8);
make_zero!(i16);
make_zero!(i32);
make_zero!(i64);
make_zero!(u8);
make_zero!(u16);
make_zero!(u32);
make_zero!(u64);
make_zero!(isize);
make_zero!(usize);

impl<T> ArgminZero for Vec<T>
where
    T: ArgminZero + Clone,
{
    #[inline]
    fn zero_like(&self) -> Vec<T> {
        if self.len() > 0 {
            vec![self[0].zero_like(); self.len()]
        } else {
            vec![T::zero(); self.len()]
        }
    }

    #[inline]
    fn zero() -> Vec<T> {
        vec![]
    }
}

#[cfg(feature = "ndarrayl")]
impl<T> ArgminZero for ndarray::Array1<T>
where
    T: Zero + ArgminZero + Clone,
{
    #[inline]
    fn zero_like(&self) -> ndarray::Array1<T> {
        ndarray::Array1::zeros(self.raw_dim())
    }

    #[inline]
    fn zero() -> ndarray::Array1<T> {
        ndarray::Array1::zeros(0)
    }
}

#[cfg(feature = "ndarrayl")]
impl<T> ArgminZero for ndarray::Array2<T>
where
    T: Zero + ArgminZero + Clone,
{
    #[inline]
    fn zero_like(&self) -> ndarray::Array2<T> {
        ndarray::Array2::zeros(self.raw_dim())
    }

    #[inline]
    fn zero() -> ndarray::Array2<T> {
        ndarray::Array2::zeros((0, 0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use paste::item;

    macro_rules! make_test {
        ($t:ty) => {
            item! {
                #[test]
                fn [<test_zero_ $t>]() {
                    let a = <$t as ArgminZero>::zero();
                    assert!(((0 as $t - a) as f64) < std::f64::EPSILON);
                }
            }

            item! {
                #[test]
                fn [<test_zero_like_ $t>]() {
                    let a = (42 as $t).zero_like();
                    assert!(((0 as $t - a) as f64) < std::f64::EPSILON);
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
