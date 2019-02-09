// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::{ArgminOp, Error};
use serde::{Deserialize, Serialize};

// #[derive(Clone, Default, Debug, Serialize, Deserialize)]
// pub struct NoOperator<T, U, H>
// where
//     T: Clone + Default + Debug + Send + Sync + Serialize + DeserializeOwned,
//     U: Clone + Default + Debug + Send + Sync + Serialize + DeserializeOwned,
//     H: Clone + Default + Debug + Send + Sync + Serialize + DeserializeOwned,
// {
//     param: std::marker::PhantomData<T>,
//     output: std::marker::PhantomData<U>,
//     hessian: std::marker::PhantomData<H>,
// }
//
// impl<T, U, H> NoOperator<T, U, H>
// where
//     T: Clone + Default + Debug + Send + Sync + Serialize + DeserializeOwned,
//     U: Clone + Default + Debug + Send + Sync + Serialize + DeserializeOwned,
//     H: Clone + Default + Debug + Send + Sync + Serialize + DeserializeOwned,
// {
//     #[allow(dead_code)]
//     pub fn new() -> Self {
//         NoOperator {
//             param: std::marker::PhantomData,
//             output: std::marker::PhantomData,
//             hessian: std::marker::PhantomData,
//         }
//     }
// }
//
// impl<T, U, H> ArgminOp for NoOperator<T, U, H>
// where
//     T: Clone + Default + Debug + Send + Sync + Serialize + DeserializeOwned,
//     U: Clone + Default + Debug + Send + Sync + Serialize + DeserializeOwned,
//     H: Clone + Default + Debug + Send + Sync + Serialize + DeserializeOwned,
// {
//     type Param = T;
//     type OperatorOutput = U;
//     type Hessian = H;
//
//     fn apply(&self, _p: &Self::Param) -> Result<Self::OperatorOutput, Error> {
//         Ok(Self::OperatorOutput::default())
//     }
//
//     fn gradient(&self, _p: &Self::Param) -> Result<Self::Param, Error> {
//         Ok(Self::Param::default())
//     }
//
//     fn hessian(&self, _p: &Self::Param) -> Result<Self::Hessian, Error> {
//         Ok(Self::Hessian::default())
//     }
//
//     fn modify(&self, _p: &Self::Param, _t: f64) -> Result<Self::Param, Error> {
//         Ok(Self::Param::default())
//     }
// }

#[derive(Clone, Default, Debug, Serialize, Deserialize)]
pub struct MinimalNoOperator {}

impl MinimalNoOperator {
    #[allow(dead_code)]
    pub fn new() -> Self {
        MinimalNoOperator {}
    }
}

impl ArgminOp for MinimalNoOperator {
    type Param = Vec<f64>;
    type Output = f64;
    type Hessian = Vec<Vec<f64>>;

    fn apply(&self, _p: &Self::Param) -> Result<Self::Output, Error> {
        unimplemented!()
    }

    fn gradient(&self, _p: &Self::Param) -> Result<Self::Param, Error> {
        unimplemented!()
    }

    fn hessian(&self, _p: &Self::Param) -> Result<Self::Hessian, Error> {
        unimplemented!()
    }

    fn modify(&self, _p: &Self::Param, _t: f64) -> Result<Self::Param, Error> {
        unimplemented!()
    }
}
