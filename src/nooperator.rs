// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::{ArgminOperator, Error};
use serde::{Deserialize, Serialize};
use std::default::Default;
use std::fmt::Debug;

#[derive(Clone, Default, Debug, Serialize, Deserialize)]
pub struct NoOperator<T, U, H>
where
    T: Clone + Default + Debug + Send + Sync,
    U: Clone + Default + Debug + Send + Sync,
    H: Clone + Default + Debug + Send + Sync,
{
    param: std::marker::PhantomData<T>,
    output: std::marker::PhantomData<U>,
    hessian: std::marker::PhantomData<H>,
}

impl<T, U, H> NoOperator<T, U, H>
where
    T: Clone + Default + Debug + Send + Sync,
    U: Clone + Default + Debug + Send + Sync,
    H: Clone + Default + Debug + Send + Sync,
{
    #[allow(dead_code)]
    pub fn new() -> Self {
        NoOperator {
            param: std::marker::PhantomData,
            output: std::marker::PhantomData,
            hessian: std::marker::PhantomData,
        }
    }
}

impl<T, U, H> ArgminOperator for NoOperator<T, U, H>
where
    T: Clone + Default + Debug + Send + Sync,
    U: Clone + Default + Debug + Send + Sync,
    H: Clone + Default + Debug + Send + Sync,
{
    type Parameters = T;
    type OperatorOutput = U;
    type Hessian = H;

    fn apply(&self, _p: &Self::Parameters) -> Result<Self::OperatorOutput, Error> {
        Ok(Self::OperatorOutput::default())
    }

    fn gradient(&self, _p: &Self::Parameters) -> Result<Self::Parameters, Error> {
        Ok(Self::Parameters::default())
    }

    fn hessian(&self, _p: &Self::Parameters) -> Result<Self::Hessian, Error> {
        Ok(Self::Hessian::default())
    }

    fn modify(&self, _p: &Self::Parameters, _t: f64) -> Result<Self::Parameters, Error> {
        Ok(Self::Parameters::default())
    }
}
