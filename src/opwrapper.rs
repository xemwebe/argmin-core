// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::{ArgminOp, Error};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct OpWrapper<O: ArgminOp> {
    op: O,
    pub cost_func_count: u64,
    pub grad_func_count: u64,
    pub hessian_func_count: u64,
    pub modify_func_count: u64,
}

impl<O: ArgminOp> OpWrapper<O> {
    pub fn new(op: &O) -> Self {
        OpWrapper {
            op: op.clone(),
            cost_func_count: 0,
            grad_func_count: 0,
            hessian_func_count: 0,
            modify_func_count: 0,
        }
    }

    pub fn apply(&mut self, param: &O::Param) -> Result<O::Output, Error> {
        self.cost_func_count += 1;
        self.op.apply(param)
    }

    pub fn gradient(&mut self, param: &O::Param) -> Result<O::Param, Error> {
        self.grad_func_count += 1;
        self.op.gradient(param)
    }

    pub fn hessian(&mut self, param: &O::Param) -> Result<O::Hessian, Error> {
        self.hessian_func_count += 1;
        self.op.hessian(param)
    }

    pub fn modify(&mut self, param: &O::Param, extent: f64) -> Result<O::Param, Error> {
        self.modify_func_count += 1;
        self.op.modify(param, extent)
    }

    pub fn consume_op<O2: ArgminOp>(&mut self, other: OpWrapper<O2>) {
        self.cost_func_count += other.cost_func_count;
        self.grad_func_count += other.grad_func_count;
        self.hessian_func_count += other.hessian_func_count;
        self.modify_func_count += other.modify_func_count;
    }
}

impl<O: ArgminOp> ArgminOp for OpWrapper<O> {
    type Param = O::Param;
    type Output = O::Output;
    type Hessian = O::Hessian;

    fn apply(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        self.op.apply(param)
    }

    fn gradient(&self, param: &Self::Param) -> Result<Self::Param, Error> {
        self.op.gradient(param)
    }

    fn hessian(&self, param: &Self::Param) -> Result<Self::Hessian, Error> {
        self.op.hessian(param)
    }

    fn modify(&self, param: &Self::Param, extent: f64) -> Result<Self::Param, Error> {
        self.op.modify(param, extent)
    }
}
