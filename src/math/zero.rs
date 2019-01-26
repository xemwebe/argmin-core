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
        vec![T::zero(); self.len()]
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
        ndarray::Array1::zeros(self.len())
    }

    #[inline]
    fn zero() -> ndarray::Array1<T> {
        ndarray::Array1::zeros(0)
    }
}
