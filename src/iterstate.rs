// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::{ArgminOp, OpWrapper};
use paste::item;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IterState<O: ArgminOp> {
    param: O::Param,
    prev_param: O::Param,
    best_param: O::Param,
    prev_best_param: O::Param,
    cost: f64,
    prev_cost: f64,
    best_cost: f64,
    prev_best_cost: f64,
    target_cost: f64,
    grad: Option<O::Param>,
    prev_grad: Option<O::Param>,
    hessian: Option<O::Hessian>,
    prev_hessian: Option<O::Hessian>,
    iter: u64,
    max_iters: u64,
    /// Number of cost function evaluations so far
    cost_func_count: u64,
    /// Number of gradient evaluations so far
    grad_func_count: u64,
    /// Number of gradient evaluations so far
    hessian_func_count: u64,
    /// Number of modify evaluations so far
    modify_func_count: u64,
}

macro_rules! setter {
    ($name:ident, $type:ty) => {
        pub fn $name(&mut self, $name: $type) -> &mut Self {
            self.$name = $name;
            self
        }
    };
}

macro_rules! getter_option {
    ($name:ident, $type:ty) => {
        item! {
            pub fn [<get_ $name>](&self) -> Option<$type> {
                self.$name.clone()
            }
        }
    };
}

macro_rules! getter {
    ($name:ident, $type:ty) => {
        item! {
            pub fn [<get_ $name>](&self) -> $type {
                self.$name.clone()
            }
        }
    };
}

impl<O: ArgminOp> IterState<O> {
    pub fn new(param: O::Param) -> Self {
        IterState {
            param: param.clone(),
            prev_param: param.clone(),
            best_param: param.clone(),
            prev_best_param: param,
            cost: std::f64::INFINITY,
            prev_cost: std::f64::INFINITY,
            best_cost: std::f64::INFINITY,
            prev_best_cost: std::f64::INFINITY,
            target_cost: std::f64::NEG_INFINITY,
            grad: None,
            prev_grad: None,
            hessian: None,
            prev_hessian: None,
            iter: 0,
            max_iters: std::u64::MAX,
            cost_func_count: 0,
            grad_func_count: 0,
            hessian_func_count: 0,
            modify_func_count: 0,
        }
    }

    pub fn param(&mut self, param: O::Param) -> &mut Self {
        std::mem::swap(&mut self.prev_param, &mut self.param);
        self.param = param;
        self
    }

    pub fn best_param(&mut self, param: O::Param) -> &mut Self {
        std::mem::swap(&mut self.prev_best_param, &mut self.best_param);
        self.best_param = param;
        self
    }

    pub fn cost(&mut self, cost: f64) -> &mut Self {
        std::mem::swap(&mut self.prev_cost, &mut self.cost);
        self.cost = cost;
        self
    }

    pub fn best_cost(&mut self, cost: f64) -> &mut Self {
        std::mem::swap(&mut self.prev_best_cost, &mut self.best_cost);
        self.best_cost = cost;
        self
    }

    pub fn grad(&mut self, grad: O::Param) -> &mut Self {
        std::mem::swap(&mut self.prev_grad, &mut self.grad);
        self.grad = Some(grad);
        self
    }

    pub fn hessian(&mut self, hessian: O::Hessian) -> &mut Self {
        std::mem::swap(&mut self.prev_hessian, &mut self.hessian);
        self.hessian = Some(hessian);
        self
    }

    setter!(target_cost, f64);
    setter!(max_iters, u64);
    getter!(param, O::Param);
    getter!(prev_param, O::Param);
    getter!(best_param, O::Param);
    getter!(prev_best_param, O::Param);
    getter!(cost, f64);
    getter!(prev_cost, f64);
    getter!(best_cost, f64);
    getter!(prev_best_cost, f64);
    getter!(target_cost, f64);
    getter!(cost_func_count, u64);
    getter!(grad_func_count, u64);
    getter!(hessian_func_count, u64);
    getter!(modify_func_count, u64);
    getter_option!(grad, O::Param);
    getter_option!(prev_grad, O::Param);
    getter_option!(hessian, O::Hessian);
    getter_option!(prev_hessian, O::Hessian);
    getter!(iter, u64);
    getter!(max_iters, u64);

    pub fn increment_iter(&mut self) {
        self.iter += 1;
    }

    pub fn increment_func_counts(&mut self, op: &OpWrapper<O>) {
        self.cost_func_count += op.cost_func_count;
        self.grad_func_count += op.grad_func_count;
        self.hessian_func_count += op.hessian_func_count;
        self.modify_func_count += op.modify_func_count;
    }

    pub fn increment_cost_func_count(&mut self, num: u64) {
        self.cost_func_count += num;
    }

    pub fn increment_grad_func_count(&mut self, num: u64) {
        self.grad_func_count += num;
    }

    pub fn increment_hessian_func_count(&mut self, num: u64) {
        self.hessian_func_count += num;
    }

    pub fn increment_modify_func_count(&mut self, num: u64) {
        self.modify_func_count += num;
    }
}
