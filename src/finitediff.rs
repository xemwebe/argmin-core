// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Finite Differentiation
//!
//! TODO: Text.

use crate::ArgminOperator;
use crate::Error;

pub trait ArgminForwardDiff
where
    Self: Sized,
{
    fn forward_diff<H>(
        &self,
        op: &ArgminOperator<Parameters = Self, OperatorOutput = f64, Hessian = H>,
    ) -> Result<Self, Error>;
}

impl ArgminForwardDiff for Vec<f64>
where
    Self: Sized,
{
    fn forward_diff<H>(
        &self,
        op: &ArgminOperator<Parameters = Self, OperatorOutput = f64, Hessian = H>,
    ) -> Result<Self, Error> {
        let fx = op.apply(&self)?;
        let n = self.len();
        Ok((0..n)
            .map(|i| {
                let mut x1 = self.clone();
                x1[i] += std::f64::EPSILON;
                let fx1 = op.apply(&x1).unwrap();
                (fx1 - fx) / std::f64::EPSILON
            })
            .collect())
    }
}
