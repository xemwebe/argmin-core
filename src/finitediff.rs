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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward_diff_vec_f64() {
        #[derive(Clone)]
        struct Problem {}

        impl ArgminOperator for Problem {
            type Parameters = Vec<f64>;
            type OperatorOutput = f64;
            type Hessian = ();

            fn apply(&self, p: &Self::Parameters) -> Result<Self::OperatorOutput, Error> {
                Ok(p[0] + p[1].powi(2))
            }

            fn gradient(&self, p: &Self::Parameters) -> Result<Self::Parameters, Error> {
                p.forward_diff(self)
            }
        }
        let prob = Problem {};
        let p = vec![0.0f64, 1.0f64];
        // println!("{:?}", prob.gradient(&p).unwrap());
        assert!((1.0 - prob.gradient(&p).unwrap()[0]).abs() < std::f64::EPSILON);
        assert!((2.0 - prob.gradient(&p).unwrap()[1]).abs() < std::f64::EPSILON);
    }
}
