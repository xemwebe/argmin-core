// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::serialization::*;
use crate::{
    ArgminCheckpoint, ArgminError, ArgminIterData, ArgminKV, ArgminLog, ArgminLogger, ArgminResult,
    ArgminWrite, ArgminWriter, Error, TerminationReason,
};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

#[derive(Serialize, Deserialize, Clone)]
pub struct OpWrapper<O: ArgminOp> {
    op: O,
    pub cost_func_count: u64,
    pub grad_func_count: u64,
    pub hessian_func_count: u64,
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

impl<O: ArgminOp> OpWrapper<O> {
    pub fn new(op: &O) -> Self {
        OpWrapper {
            op: op.clone(),
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

    // /// Set the counts to zero
    // pub fn reset(&mut self) {
    //     self.cost_func_count = 0;
    //     self.grad_func_count = 0;
    //     self.hessian_func_count = 0;
    // }
}

/// This trait needs to be implemented for every operator/cost function.
///
/// It is required to implement the `apply` method, all others are optional and provide a default
/// implementation which is essentially returning an error which indicates that the method has not
/// been implemented. Those methods (`gradient` and `modify`) only need to be implemented if the
/// uses solver requires it.
pub trait ArgminOp: Clone + Send + Sync + Serialize {
    /// Type of the parameter vector
    type Param: Clone + Serialize;
    /// Output of the operator
    type Output;
    /// Type of Hessian
    type Hessian: Clone + Serialize;

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

// currently this uses owned values, ideally this would only be references, but I don't want to
// fight the borrow checker right now.
pub struct IterState<P, H> {
    pub cur_param: P,
    pub best_param: P,
    pub cur_cost: f64,
    pub best_cost: f64,
    pub target_cost: f64,
    pub cur_grad: P,
    pub cur_hessian: H,
    pub cur_iter: u64,
    pub max_iters: u64,
}

pub trait Solver<O: ArgminOp>: Serialize {
    /// Computes one iteration of the algorithm.
    fn next_iter(
        &mut self,
        op: &mut OpWrapper<O>,
        state: IterState<O::Param, O::Hessian>,
    ) -> Result<ArgminIterData<O::Param, O::Param>, Error>;

    /// Initializes the algorithm
    ///
    /// This is executed before any iterations are performed. It can be used to perform
    /// precomputations. The default implementation corresponds to doing nothing.
    fn init(
        &mut self,
        _op: &mut OpWrapper<O>,
    ) -> Result<Option<ArgminIterData<O::Param, O::Param>>, Error> {
        Ok(None)
    }

    fn terminate_internal(&mut self, state: &IterState<O::Param, O::Hessian>) -> TerminationReason {
        let solver_terminate = self.terminate(state);
        if solver_terminate.terminated() {
            return solver_terminate;
        }
        if state.cur_iter >= state.max_iters {
            return TerminationReason::MaxItersReached;
        }
        if state.cur_cost <= state.target_cost {
            return TerminationReason::TargetCostReached;
        }
        TerminationReason::NotTerminated
    }

    fn terminate(&mut self, _state: &IterState<O::Param, O::Hessian>) -> TerminationReason {
        TerminationReason::NotTerminated
    }
}

#[derive(Serialize, Deserialize)]
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
    // TODO: make getters for these values
    /// Number of cost function evaluations so far
    pub cost_func_count: u64,
    /// Number of gradient evaluations so far
    pub grad_func_count: u64,
    /// Number of gradient evaluations so far
    pub hessian_func_count: u64,
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

    pub fn from_checkpoint<P: AsRef<Path>>(path: P) -> Result<Self, Error>
    where
        Self: Sized + DeserializeOwned,
    {
        load_checkpoint(path)
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

        let mut op_wrapper = OpWrapper::new(&self.op);
        let init_data = self.solver.init(&mut op_wrapper)?;

        // If init() returned something, deal with it
        if let Some(data) = init_data {
            // Set new current parameter
            self.cur_param = data.param();
            self.cur_cost = data.cost();
            // check if parameters are the best so far
            if self.cur_cost <= self.best_cost {
                self.best_param = self.cur_param.clone();
                self.best_cost = self.cur_cost;
            }
        }

        // TODO: write a method for this?
        self.cost_func_count = op_wrapper.cost_func_count;
        self.grad_func_count = op_wrapper.grad_func_count;
        self.hessian_func_count = op_wrapper.hessian_func_count;

        while running.load(Ordering::SeqCst) {
            let state = self.to_state();

            // check first if it has already terminated
            // This should probably be solved better.
            // First, check if it isn't already terminated. If it isn't, evaluate the
            // stopping criteria. If `self.terminate()` is called without the checking
            // whether it has terminated already, then it may overwrite a termination set
            // within `next_iter()`!
            if !self.termination_reason.terminated() {
                self.termination_reason = self.solver.terminate_internal(&state);
            }
            // Now check once more if the algorithm has terminated. If yes, then break.
            if self.termination_reason.terminated() {
                break;
            }

            // Start time measurement
            let start = std::time::Instant::now();

            let data = self.solver.next_iter(&mut op_wrapper, state)?;

            self.cost_func_count = op_wrapper.cost_func_count;
            self.grad_func_count = op_wrapper.grad_func_count;
            self.hessian_func_count = op_wrapper.hessian_func_count;

            // End time measurement
            let duration = start.elapsed();

            // Set new current parameter
            self.cur_param = data.param();
            self.cur_cost = data.cost();

            // check if parameters are the best so far
            if self.cur_cost <= self.best_cost {
                self.best_param = self.cur_param.clone();
                self.best_cost = self.cur_cost;
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

            self.checkpoint.store_cond(self, self.cur_iter)?;

            // Check if termination occured inside next_iter()
            let iter_term = data.termination_reason();
            if iter_term.terminated() {
                self.termination_reason = iter_term;
                break;
            }
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

    pub fn run_fast(&mut self) -> Result<ArgminResult<O::Param>, Error> {
        let mut op_wrapper = OpWrapper::new(&self.op);
        let init_data = self.solver.init(&mut op_wrapper)?;

        // If init() returned something, deal with it
        if let Some(data) = init_data {
            // Set new current parameter
            self.cur_param = data.param();
            self.cur_cost = data.cost();
            // check if parameters are the best so far
            if self.cur_cost <= self.best_cost {
                self.best_param = self.cur_param.clone();
                self.best_cost = self.cur_cost;
            }
        }

        // TODO: write a method for this?
        self.cost_func_count = op_wrapper.cost_func_count;
        self.grad_func_count = op_wrapper.grad_func_count;
        self.hessian_func_count = op_wrapper.hessian_func_count;

        loop {
            let state = self.to_state();

            // check first if it has already terminated
            // This should probably be solved better.
            // First, check if it isn't already terminated. If it isn't, evaluate the
            // stopping criteria. If `self.terminate()` is called without the checking
            // whether it has terminated already, then it may overwrite a termination set
            // within `next_iter()`!
            if !self.termination_reason.terminated() {
                self.termination_reason = self.solver.terminate_internal(&state);
            }
            // Now check once more if the algorithm has terminated. If yes, then break.
            if self.termination_reason.terminated() {
                break;
            }

            let data = self.solver.next_iter(&mut op_wrapper, state)?;

            self.cost_func_count = op_wrapper.cost_func_count;
            self.grad_func_count = op_wrapper.grad_func_count;
            self.hessian_func_count = op_wrapper.hessian_func_count;

            // Set new current parameter
            self.cur_param = data.param();
            self.cur_cost = data.cost();

            // check if parameters are the best so far
            if self.cur_cost <= self.best_cost {
                self.best_param = self.cur_param.clone();
                self.best_cost = self.cur_cost;
                self.writer.write(&self.best_param, self.cur_iter, true)?;
            }

            // increment iteration number
            self.cur_iter += 1;

            self.checkpoint.store_cond(self, self.cur_iter)?;

            // Check if termination occured inside next_iter()
            let iter_term = data.termination_reason();
            if iter_term.terminated() {
                self.termination_reason = iter_term;
                break;
            }
        }

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

    pub fn to_state(&self) -> IterState<O::Param, O::Hessian> {
        IterState {
            cur_param: self.cur_param.clone(),
            best_param: self.best_param.clone(),
            cur_cost: self.cur_cost,
            best_cost: self.best_cost,
            target_cost: self.target_cost,
            cur_grad: self.cur_grad.clone(),
            cur_hessian: self.cur_hessian.clone(),
            cur_iter: self.cur_iter,
            max_iters: self.max_iters,
        }
    }

    /// Set checkpoint directory
    pub fn checkpoint_dir(mut self, dir: &str) -> Self {
        self.checkpoint.set_dir(dir);
        self
    }

    /// Set checkpoint name
    pub fn checkpoint_name(mut self, dir: &str) -> Self {
        self.checkpoint.set_name(dir);
        self
    }

    pub fn checkpoint_mode(mut self, mode: CheckpointMode) -> Self {
        self.checkpoint.set_mode(mode);
        self
    }
}
