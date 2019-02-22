// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Storage for basic information most solvers require.
//!
//! The aim is to provide a common interface in order to make code generation and the
//! implementation of solvers easier and more standardized. This is likely to change a lot over
//! time and will require quite some time till it is stabilized.
//! Also, the name is probably not the best choice.
//!
//! TODO:
//!   * It would be great if `T` did not have to implement `Default`

use crate::logging::ArgminLogger;
use crate::output::ArgminWriter;
use crate::termination::TerminationReason;
use crate::ArgminKV;
use crate::ArgminLog;
use crate::ArgminOp;
use crate::ArgminResult;
use crate::ArgminWrite;
use crate::Error;
use crate::{ArgminCheckpoint, CheckpointMode};
use serde::{Deserialize, Serialize};
use std;
use std::default::Default;
use std::sync::Arc;

/// Storage for data needed by most solvers
///
/// TODO: cur_cost, best_cost and target_cost should be `U`, but then initialization is difficult
/// as it cannot be expected that each `U` has something like `INFINITY` and `NEG_INFINITY`...
#[derive(Clone, Serialize, Deserialize)]
pub struct ArgminBase<O: ArgminOp> {
    /// The operator/cost function
    operator: O,

    /// Current parameter vector
    cur_param: O::Param,

    /// Current best parameter vector
    best_param: O::Param,

    /// Current cost function value
    cur_cost: f64,

    /// Cost function value of current best parameter vector
    best_cost: f64,

    /// Target cost function value
    target_cost: f64,

    /// Current gradient
    cur_grad: O::Param,

    /// Current hessian
    cur_hessian: O::Hessian,

    /// Current iteration number
    cur_iter: u64,

    /// Maximum number of iterations
    max_iters: u64,

    /// Number of cost function evaluations so far
    cost_func_count: u64,

    /// Number of gradient evaluations so far
    grad_func_count: u64,

    /// Number of gradient evaluations so far
    hessian_func_count: u64,

    /// Reason of termination
    termination_reason: TerminationReason,

    /// Total time the solver required.
    total_time: std::time::Duration,

    /// Storage for loggers
    #[serde(skip)]
    logger: ArgminLogger,

    /// Storage for writers
    #[serde(skip)]
    writer: ArgminWriter<O::Param>,

    /// Checkpoint
    checkpoint: ArgminCheckpoint,
}

impl<O> ArgminBase<O>
where
    O: ArgminOp,
{
    /// Constructor
    pub fn new(operator: O, param: O::Param) -> Self {
        ArgminBase {
            operator,
            cur_param: param.clone(),
            best_param: param,
            cur_cost: std::f64::INFINITY,
            best_cost: std::f64::INFINITY,
            target_cost: std::f64::NEG_INFINITY,
            cur_grad: O::Param::default(),
            cur_hessian: O::Hessian::default(),
            cur_iter: 0,
            max_iters: std::u64::MAX,
            cost_func_count: 0,
            grad_func_count: 0,
            hessian_func_count: 0,
            termination_reason: TerminationReason::NotTerminated,
            total_time: std::time::Duration::new(0, 0),
            logger: ArgminLogger::new(),
            writer: ArgminWriter::new(),
            checkpoint: ArgminCheckpoint::default(),
        }
    }

    /// Return the KV for the initial logging
    pub fn kv_for_logs(&self) -> ArgminKV {
        make_kv!(
            "target_cost" => self.target_cost;
            "max_iters" => self.max_iters;
            "termination_reason" => self.termination_reason;
        )
    }

    /// Return the KV for logging of the iterations
    pub fn kv_for_iter(&self) -> ArgminKV {
        make_kv!(
            "iter" => self.cur_iter;
            "best_cost" => self.best_cost;
            "cur_cost" => self.cur_cost;
            "cost_func_count" => self.cost_func_count;
            "grad_func_count" => self.grad_func_count;
            "hessian_func_count" => self.hessian_func_count;
        )
    }

    /// Reset `self` to its initial state.
    ///
    /// This is dangerous the way it is implemented right now. This has to be done better. For
    /// instance, all old data needs to be kept in order to be able to actually go back to the
    /// initial state. Also, this method definitely needs to be kept up to date otherwise nasty bugs
    /// may happen.
    pub fn reset(&mut self) {
        self.cur_iter = 0;
        self.cur_cost = std::f64::INFINITY;
        self.best_cost = std::f64::INFINITY;
        self.cost_func_count = 0;
        self.grad_func_count = 0;
        self.hessian_func_count = 0;
        self.termination_reason = TerminationReason::NotTerminated;
        self.total_time = std::time::Duration::new(0, 0);
    }

    /// Apply the operator to `param`
    pub fn apply(&mut self, param: &O::Param) -> Result<O::Output, Error> {
        self.increment_cost_func_count();
        self.operator.apply(param)
    }

    /// Compute the gradient at `param`
    pub fn gradient(&mut self, param: &O::Param) -> Result<O::Param, Error> {
        self.increment_grad_func_count();
        self.operator.gradient(param)
    }

    /// Compute the hessian at `param`
    pub fn hessian(&mut self, param: &O::Param) -> Result<O::Hessian, Error> {
        self.increment_hessian_func_count();
        self.operator.hessian(param)
    }

    /// Modify a `param` with the `modify` method of `operator`.
    pub fn modify(&self, param: &O::Param, factor: f64) -> Result<O::Param, Error> {
        self.operator.modify(&param, factor)
    }

    /// Set the current parameter vector
    pub fn set_cur_param(&mut self, param: O::Param) -> &mut Self {
        self.cur_param = param;
        self
    }

    /// Return the current parameter vector
    pub fn cur_param(&self) -> O::Param {
        self.cur_param.clone()
    }

    /// Set the new best parameter vector
    pub fn set_best_param(&mut self, param: O::Param) -> &mut Self {
        self.best_param = param;
        // FIXME: Remove unwrap!
        self.writer
            .write(&self.best_param, self.cur_iter(), true)
            .unwrap();
        self
    }

    /// Return the current best parameter vector
    pub fn best_param(&self) -> O::Param {
        self.best_param.clone()
    }

    /// Set the current cost function value
    pub fn set_cur_cost(&mut self, cost: f64) -> &mut Self {
        self.cur_cost = cost;
        self
    }

    /// Return the current cost function value
    pub fn cur_cost(&self) -> f64 {
        self.cur_cost
    }

    /// Set the cost function value of the current best parameter vector
    pub fn set_best_cost(&mut self, cost: f64) -> &mut Self {
        self.best_cost = cost;
        self
    }

    /// Return the cost function value of the current best parameter vector
    pub fn best_cost(&self) -> f64 {
        self.best_cost
    }

    /// Set the current gradient
    pub fn set_cur_grad(&mut self, grad: O::Param) -> &mut Self {
        self.cur_grad = grad;
        self
    }

    /// Return the current gradient
    pub fn cur_grad(&self) -> O::Param {
        self.cur_grad.clone()
    }

    /// Set the current hessian
    pub fn set_cur_hessian(&mut self, hessian: O::Hessian) -> &mut Self {
        self.cur_hessian = hessian;
        self
    }

    /// Return the current hessian
    pub fn cur_hessian(&self) -> O::Hessian {
        self.cur_hessian.clone()
    }

    /// Set the target cost function value
    pub fn set_target_cost(&mut self, cost: f64) -> &mut Self {
        self.target_cost = cost;
        self
    }

    /// Return the target cost function value
    pub fn target_cost(&self) -> f64 {
        self.target_cost
    }

    /// Increment the number of iterations.
    pub fn increment_iter(&mut self) -> &mut Self {
        self.cur_iter += 1;
        self
    }

    /// Return the current number of iterations
    pub fn cur_iter(&self) -> u64 {
        self.cur_iter
    }

    /// Increment the cost function evaluation count
    pub fn increment_cost_func_count(&mut self) -> &mut Self {
        self.cost_func_count += 1;
        self
    }

    /// Increaese the cost function evaluation count by a given value
    pub fn increase_cost_func_count(&mut self, count: u64) -> &mut Self {
        self.cost_func_count += count;
        self
    }

    /// Return the cost function evaluation count
    pub fn cost_func_count(&self) -> u64 {
        self.cost_func_count
    }

    /// Increment the gradient evaluation count
    pub fn increment_grad_func_count(&mut self) -> &mut Self {
        self.grad_func_count += 1;
        self
    }

    /// Increase the gradient evaluation count by a given value
    pub fn increase_grad_func_count(&mut self, count: u64) -> &mut Self {
        self.grad_func_count += count;
        self
    }

    /// Return the gradient evaluation count
    pub fn grad_func_count(&self) -> u64 {
        self.grad_func_count
    }

    /// Increment the hessian evaluation count
    pub fn increment_hessian_func_count(&mut self) -> &mut Self {
        self.hessian_func_count += 1;
        self
    }

    /// Increase the hessian evaluation count by a given value
    pub fn increase_hessian_func_count(&mut self, count: u64) -> &mut Self {
        self.hessian_func_count += count;
        self
    }

    /// Return the gradient evaluation count
    pub fn hessian_func_count(&self) -> u64 {
        self.hessian_func_count
    }

    /// Set the maximum number of iterations.
    pub fn set_max_iters(&mut self, iters: u64) -> &mut Self {
        self.max_iters = iters;
        self
    }

    /// Return the maximum number of iterations
    pub fn max_iters(&self) -> u64 {
        self.max_iters
    }

    /// Set the `TerminationReason`
    pub fn set_termination_reason(&mut self, reason: TerminationReason) -> &mut Self {
        self.termination_reason = reason;
        self
    }

    /// Return the `TerminationReason`
    pub fn termination_reason(&self) -> TerminationReason {
        self.termination_reason.clone()
    }

    /// Return the textual representation of the `TerminationReason`
    pub fn termination_reason_text(&self) -> &str {
        self.termination_reason.text()
    }

    /// Return whether the algorithm has terminated or not
    pub fn terminated(&self) -> bool {
        self.termination_reason.terminated()
    }

    /// Return the result.
    pub fn result(&self) -> ArgminResult<O::Param> {
        ArgminResult::new(
            self.best_param.clone(),
            self.best_cost(),
            self.cur_iter(),
            self.termination_reason(),
        )
    }

    /// Set the total time needed by the solver
    pub fn set_total_time(&mut self, time: std::time::Duration) -> &mut Self {
        self.total_time = time;
        self
    }

    /// Return the total time
    pub fn total_time(&self) -> std::time::Duration {
        self.total_time
    }

    /// Add a logger to the list of loggers
    pub fn add_logger(&mut self, logger: Arc<ArgminLog>) -> &mut Self {
        self.logger.push(logger);
        self
    }

    /// Add a writer to the list of writers
    pub fn add_writer(&mut self, writer: Arc<ArgminWrite<Param = O::Param>>) -> &mut Self {
        self.writer.push(writer);
        self
    }

    /// Log a `kv`
    pub fn log_iter(&self, kv: &ArgminKV) -> Result<(), Error> {
        self.logger.log_iter(kv)
    }

    /// Log a message and a `kv`
    pub fn log_info(&self, msg: &str, kv: &ArgminKV) -> Result<(), Error> {
        self.logger.log_info(msg, kv)
    }

    /// Write
    pub fn write(&self, param: &O::Param) -> Result<(), Error> {
        self.writer.write(param, self.cur_iter(), false)
    }

    /// Set checkpoint directory
    pub fn set_checkpoint_dir(&mut self, dir: &str) {
        self.checkpoint.set_dir(dir);
    }

    /// Set checkpoint name
    pub fn set_checkpoint_name(&mut self, dir: &str) {
        self.checkpoint.set_name(dir);
    }

    pub fn set_checkpoint_mode(&mut self, mode: CheckpointMode) {
        self.checkpoint.set_mode(mode);
    }

    pub fn store_checkpoint<T: Serialize>(&self, solver: &T) -> Result<(), Error> {
        self.checkpoint.store_cond(solver, self.cur_iter())
    }
}

impl<O> std::fmt::Debug for ArgminBase<O>
where
    O: ArgminOp,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        writeln!(f, "ArgminBase:")?;
        writeln!(f, "   cur_cost:           {}", self.cur_cost)?;
        writeln!(f, "   best_cost:          {}", self.best_cost)?;
        writeln!(f, "   target_cost:        {}", self.target_cost)?;
        writeln!(f, "   cur_iter:           {}", self.cur_iter)?;
        writeln!(f, "   max_iter:           {}", self.max_iters)?;
        writeln!(f, "   cost_func_count:    {}", self.cost_func_count)?;
        writeln!(f, "   grad_func_count:    {}", self.grad_func_count)?;
        writeln!(f, "   hessian_func_count: {}", self.hessian_func_count)?;
        writeln!(f, "   termination_reason: {}", self.termination_reason)?;
        writeln!(f, "   total_time:         {:?}", self.total_time)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    send_sync_test!(
        argmin_base,
        ArgminBase<crate::nooperator::MinimalNoOperator>
    );
}
