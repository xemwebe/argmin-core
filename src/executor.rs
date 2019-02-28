// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::{
    ArgminCheckpoint, ArgminError, ArgminIterData, ArgminKV, ArgminLog, ArgminLogger, ArgminResult,
    ArgminWrite, ArgminWriter, Error, TerminationReason,
};
// use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

pub struct OpWrapper<'a, O: ArgminOp> {
    op: &'a O,
    pub cost_func_count: u64,
    pub grad_func_count: u64,
    pub hessian_func_count: u64,
}

impl<'a, O: ArgminOp> OpWrapper<'a, O> {
    pub fn new(op: &'a O) -> Self {
        OpWrapper {
            op: op,
            cost_func_count: 0,
            grad_func_count: 0,
            hessian_func_count: 0,
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
}

pub trait ArgminOp: Send + Sync + erased_serde::Serialize {
    /// Type of the parameter vector
    type Param;
    /// Output of the operator. Most solvers expect `f64`.
    type Output;
    /// Type of Hessian
    type Hessian;

    /// Applies the operator/cost function to parameters
    fn apply(&self, _param: &Self::Param) -> Result<Self::Output, Error> {
        Err(ArgminError::NotImplemented {
            text: "Method `apply` of ArgminOp trait not implemented!".to_string(),
        }
        .into())
    }

    /// Computes the gradient at the given parameters
    fn gradient(&self, _param: &Self::Param) -> Result<Self::Param, Error> {
        Err(ArgminError::NotImplemented {
            text: "Method `gradient` of ArgminOp trait not implemented!".to_string(),
        }
        .into())
    }

    /// Computes the hessian at the given parameters
    fn hessian(&self, _param: &Self::Param) -> Result<Self::Hessian, Error> {
        Err(ArgminError::NotImplemented {
            text: "Method `hessian` of ArgminOp trait not implemented!".to_string(),
        }
        .into())
    }

    /// Modifies a parameter vector. Comes with a variable that indicates the "degree" of the
    /// modification.
    fn modify(&self, _param: &Self::Param, _extent: f64) -> Result<Self::Param, Error> {
        Err(ArgminError::NotImplemented {
            text: "Method `modify` of ArgminOp trait not implemented!".to_string(),
        }
        .into())
    }
}

pub trait Solver<O: ArgminOp> {
    /// Computes one iteration of the algorithm.
    fn next_iter<'a>(
        &mut self,
        op: &mut OpWrapper<'a, O>,
        cur_param: &O::Param,
    ) -> Result<ArgminIterData<O::Param>, Error>;

    /// Initializes the algorithm
    ///
    /// This is executed before any iterations are performed. It can be used to perform
    /// precomputations. The default implementation corresponds to doing nothing.
    fn init(&mut self) -> Result<(), Error> {
        Ok(())
    }

    fn terminate(&mut self) -> TerminationReason {
        TerminationReason::NotTerminated
    }
}

// #[derive(Serialize, Deserialize)]
pub struct Executor<O: ArgminOp, S> {
    /// solver
    solver: S,
    /// operator
    op: O,
    /// Current parameter vector
    cur_param: O::Param,
    /// Current best parameter vector
    best_param: O::Param,
    /// Current cost function value
    cur_cost: f64,
    /// Previous cost function value
    prev_cost: f64,
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
    // #[serde(skip)]
    logger: ArgminLogger,
    /// Storage for writers
    // #[serde(skip)]
    writer: ArgminWriter<O::Param>,
    /// Checkpoint
    checkpoint: ArgminCheckpoint,
}

impl<O, S> Executor<O, S>
where
    O: ArgminOp,
    O::Param: Clone + Default,
    O::Hessian: Default,
    S: Solver<O>,
{
    pub fn new(op: O, solver: S, init_param: O::Param) -> Self {
        Executor {
            solver: solver,
            op: op,
            cur_param: init_param.clone(),
            best_param: init_param,
            cur_cost: std::f64::INFINITY,
            prev_cost: std::f64::INFINITY,
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

    fn apply(&mut self, param: &O::Param) -> Result<O::Output, Error> {
        self.cost_func_count += 1;
        self.op.apply(param)
    }

    pub fn run(&mut self) -> Result<ArgminResult<O::Param>, Error> {
        let total_time = std::time::Instant::now();

        // do the inital logging
        // let logs = make_kv!("max_iters" => self.max_iters();
        //                     #(#logs_str => #logs_expr;)*);
        let logs = make_kv!("max_iters" => self.max_iters;);
        // self.base.log_info(#solver_name, &logs)?;
        self.logger.log_info("blah", &logs)?;

        let running = Arc::new(AtomicBool::new(true));

        #[cfg(feature = "ctrlc")]
        {
            // Set up the Ctrl-C handler
            use ctrlc;
            let r = running.clone();
            // This is currently a hack to allow checkpoints to be run again within the
            // same program (usually not really a usecase anyway). Unfortunately, this
            // means that any subsequent run started afterwards will have not Ctrl-C
            // handling available... This should also be a problem in case one tries to run
            // two consecutive optimizations. There is ongoing work in the ctrlc crate
            // (channels and such) which may solve this problem. So far, we have to live
            // with this.
            match ctrlc::set_handler(move || {
                r.store(false, Ordering::SeqCst);
            }) {
                Err(ctrlc::Error::MultipleHandlers) => Ok(()),
                Err(e) => Err(e),
                Ok(r) => Ok(r),
            }?;
        }

        self.solver.init()?;

        while running.load(Ordering::SeqCst) {
            // check first if it has already terminated
            // This should probably be solved better.
            // First, check if it isn't already terminated. If it isn't, evaluate the
            // stopping criteria. If `self.terminate()` is called without the checking
            // whether it has terminated already, then it may overwrite a termination set
            // within `next_iter()`!
            if !self.termination_reason.terminated() {
                self.solver.terminate();
            }
            // Now check once more if the algorithm has terminated. If yes, then break.
            if self.termination_reason.terminated() {
                break;
            }

            // Start time measurement
            let start = std::time::Instant::now();

            let mut op_wrapper = OpWrapper::new(&self.op);

            let data = self.solver.next_iter(&mut op_wrapper, &self.cur_param)?;

            self.cost_func_count += op_wrapper.cost_func_count;
            self.grad_func_count += op_wrapper.grad_func_count;
            self.hessian_func_count += op_wrapper.hessian_func_count;

            // End time measurement
            let duration = start.elapsed();

            // Set new current parameter
            self.cur_param = data.param();
            self.cur_cost = data.cost();

            // check if parameters are the best so far
            if data.cost() <= self.best_cost {
                self.best_param = data.param();
                self.cur_cost = data.cost();
                self.writer.write(&self.best_param, self.cur_iter, true)?;
            }

            // logging
            let mut log = make_kv!(
                "iter" => self.cur_iter;
                "best_cost" => self.best_cost;
                "cur_cost" => self.cur_cost;
                "cost_func_count" => self.cost_func_count;
                "grad_func_count" => self.grad_func_count;
                "hessian_func_count" => self.hessian_func_count;
            );
            if let Some(ref mut iter_log) = data.get_kv() {
                iter_log.push(
                    "time",
                    duration.as_secs() as f64 + duration.subsec_nanos() as f64 * 1e-9,
                );
                log.merge(&mut iter_log.clone());
            }
            self.logger.log_iter(&log)?;

            // Write to file or something
            self.writer.write(&self.cur_param, self.cur_iter, false)?;

            // increment iteration number
            self.cur_iter += 1;

            // TODO TODO TODO
            // self.base.store_checkpoint(&self)?;
        }

        // in case it stopped prematurely and `termination_reason` is still `NotTerminated`,
        // someone must have pulled the handbrake
        if self.cur_iter < self.max_iters && !self.termination_reason.terminated() {
            self.termination_reason = TerminationReason::Aborted;
        }

        self.total_time = total_time.elapsed();

        let kv = make_kv!(
            "termination_reason" => self.termination_reason;
            "total_time" => self.total_time.as_secs() as f64 +
                            self.total_time.subsec_nanos() as f64 * 1e-9;
        );

        self.logger.log_info(
            &format!(
                "Terminated: {reason}",
                reason = self.termination_reason.text(),
            ),
            &kv,
        )?;

        Ok(ArgminResult::new(
            self.best_param.clone(),
            self.best_cost,
            self.cur_iter,
            self.termination_reason,
        ))
    }

    /// Attaches a logger which implements `ArgminLog` to the solver.
    pub fn add_logger(mut self, logger: std::sync::Arc<ArgminLog>) -> Self {
        self.logger.push(logger);
        self
    }

    pub fn set_max_iters(mut self, iters: u64) -> Self {
        self.max_iters = iters;
        self
    }
}