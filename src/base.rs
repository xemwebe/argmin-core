// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use logging::ArgminLogger;
use output::ArgminWriter;
use std;
use termination::TerminationReason;
use ArgminKV;
use ArgminLog;
use ArgminResult;
use ArgminWrite;
use Error;

pub struct ArgminBase<T> {
    cur_param: T,
    best_param: T,
    cur_cost: f64,
    best_cost: f64,
    cur_iter: u64,
    max_iters: u64,
    cost_func_count: u64,
    grad_func_count: u64,
    termination_reason: TerminationReason,
    total_time: u64,
    logger: ArgminLogger,
    writer: ArgminWriter<T>,
}

impl<T: Clone> ArgminBase<T> {
    pub fn new(param: T) -> Self {
        ArgminBase {
            cur_param: param.clone(),
            best_param: param,
            cur_cost: std::f64::NAN,
            best_cost: std::f64::NAN,
            cur_iter: 0,
            max_iters: std::u64::MAX,
            cost_func_count: 0,
            grad_func_count: 0,
            termination_reason: TerminationReason::NotTerminated,
            total_time: 0,
            logger: ArgminLogger::new(),
            writer: ArgminWriter::new(),
        }
    }

    pub fn set_cur_param(&mut self, param: T) -> &mut Self {
        self.cur_param = param;
        self
    }

    pub fn cur_param(&self) -> T {
        self.cur_param.clone()
    }

    pub fn set_best_param(&mut self, param: T) -> &mut Self {
        self.best_param = param;
        self
    }

    pub fn best_param(&self) -> T {
        self.best_param.clone()
    }

    pub fn set_cur_cost(&mut self, cost: f64) -> &mut Self {
        self.cur_cost = cost;
        self
    }

    pub fn cur_cost(&self) -> f64 {
        self.cur_cost
    }

    pub fn set_best_cost(&mut self, cost: f64) -> &mut Self {
        self.best_cost = cost;
        self
    }

    pub fn best_cost(&self) -> f64 {
        self.best_cost
    }

    pub fn increment_iter(&mut self) -> &mut Self {
        self.cur_iter += 1;
        self
    }

    pub fn cur_iter(&self) -> u64 {
        self.cur_iter
    }

    pub fn increment_cost_func_count(&mut self) -> &mut Self {
        self.cost_func_count += 1;
        self
    }

    pub fn cost_func_count(&self) -> u64 {
        self.cost_func_count
    }

    pub fn increment_grad_func_count(&mut self) -> &mut Self {
        self.grad_func_count += 1;
        self
    }

    pub fn grad_func_count(&self) -> u64 {
        self.grad_func_count
    }

    pub fn set_max_iters(&mut self, iters: u64) -> &mut Self {
        self.max_iters = iters;
        self
    }

    pub fn max_iters(&mut self) -> u64 {
        self.max_iters
    }

    pub fn set_termination_reason(&mut self, reason: TerminationReason) -> &mut Self {
        self.termination_reason = reason;
        self
    }

    pub fn termination_reason(&self) -> TerminationReason {
        self.termination_reason.clone()
    }

    pub fn termination_reason_test(&self) -> &str {
        self.termination_reason.text()
    }

    pub fn result(&self) -> ArgminResult<T> {
        ArgminResult::new(
            self.best_param.clone(),
            self.best_cost(),
            self.cur_iter(),
            self.termination_reason(),
        )
    }

    pub fn set_total_time(&mut self, time: u64) -> &mut Self {
        self.total_time = time;
        self
    }

    pub fn add_logger(&mut self, logger: Box<ArgminLog>) -> &mut Self {
        self.logger.push(logger);
        self
    }

    pub fn add_writer(&mut self, writer: Box<ArgminWrite<Param = T>>) -> &mut Self {
        self.writer.push(writer);
        self
    }

    pub fn log_iter(&self, kv: &ArgminKV) -> Result<(), Error> {
        self.logger.log_iter(kv)
    }

    pub fn log_info(&self, msg: &str, kv: &ArgminKV) -> Result<(), Error> {
        self.logger.log_info(msg, kv)
    }
}
