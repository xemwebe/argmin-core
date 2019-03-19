// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

// TODO: Logging of "initial info"

use crate::serialization::*;
use crate::{
    ArgminCheckpoint, ArgminIterData, ArgminKV, ArgminOp, ArgminResult, Error, IterState, Observe,
    Observer, ObserverMode, OpWrapper, Solver, TerminationReason,
};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

#[derive(Serialize, Deserialize)]
pub struct Executor<O: ArgminOp, S> {
    /// solver
    solver: S,
    /// operator
    op: O,
    /// State
    state: IterState<O>,
    /// Storage for observers
    #[serde(skip)]
    observers: Observer<O>,
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
        let state = IterState::new(init_param);
        Executor {
            solver,
            op,
            state,
            observers: Observer::new(),
            checkpoint: ArgminCheckpoint::default(),
        }
    }

    pub fn from_checkpoint<P: AsRef<Path>>(path: P) -> Result<Self, Error>
    where
        Self: Sized + DeserializeOwned,
    {
        load_checkpoint(path)
    }

    fn update(&mut self, data: &ArgminIterData<O>) -> Result<(), Error> {
        if let Some(cur_param) = data.get_param() {
            self.state.param(cur_param);
        }
        if let Some(cur_cost) = data.get_cost() {
            self.state.cost(cur_cost);
        }
        // check if parameters are the best so far
        if self.state.get_cost() <= self.state.get_best_cost() {
            let param = self.state.get_param().clone();
            let cost = self.state.get_cost();
            self.state.best_param(param).best_cost(cost);
            self.state.new_best();
        }
        if let Some(grad) = data.get_grad() {
            self.state.grad(grad);
        }
        if let Some(hessian) = data.get_hessian() {
            self.state.hessian(hessian);
        }
        if let Some(termination_reason) = data.get_termination_reason() {
            self.state.termination_reason(termination_reason);
        }
        Ok(())
    }

    pub fn run(mut self) -> Result<ArgminResult<O>, Error> {
        let total_time = std::time::Instant::now();

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
                r => r,
            }?;
        }

        let mut op_wrapper = OpWrapper::new(&self.op);
        let init_data = self.solver.init(&mut op_wrapper, &self.state)?;

        let mut logs = make_kv!("max_iters" => self.state.get_max_iters(););

        // If init() returned something, deal with it
        if let Some(data) = init_data {
            self.update(&data)?;
            logs = logs.merge(&mut data.get_kv());
        }

        // Observe after init
        self.observers.observe_init(S::NAME, &logs)?;

        self.state.increment_func_counts(&op_wrapper);

        while running.load(Ordering::SeqCst) {
            // check first if it has already terminated
            // This should probably be solved better.
            // First, check if it isn't already terminated. If it isn't, evaluate the
            // stopping criteria. If `self.terminate()` is called without the checking
            // whether it has terminated already, then it may overwrite a termination set
            // within `next_iter()`!
            if !self.state.terminated() {
                self.state
                    .termination_reason(self.solver.terminate_internal(&self.state));
            }
            // Now check once more if the algorithm has terminated. If yes, then break.
            if self.state.terminated() {
                break;
            }

            // Start time measurement
            let start = std::time::Instant::now();

            let data = self.solver.next_iter(&mut op_wrapper, &self.state)?;

            self.state.increment_func_counts(&op_wrapper);

            // End time measurement
            let duration = start.elapsed();

            self.update(&data)?;

            let log = data.get_kv().merge(&mut make_kv!(
                "time" => duration.as_secs() as f64 + f64::from(duration.subsec_nanos()) * 1e-9;
            ));

            self.observers.observe_iter(&self.state, &log)?;

            // increment iteration number
            self.state.increment_iter();

            self.checkpoint.store_cond(&self, self.state.get_iter())?;

            self.state.time(total_time.elapsed());

            // Check if termination occured inside next_iter()
            if self.state.terminated() {
                break;
            }
        }

        // in case it stopped prematurely and `termination_reason` is still `NotTerminated`,
        // someone must have pulled the handbrake
        if self.state.get_iter() < self.state.get_max_iters() && !self.state.terminated() {
            self.state.termination_reason(TerminationReason::Aborted);
        }

        Ok(ArgminResult::new(
            self.state.get_best_param(),
            self.state.get_best_cost(),
            self.state.get_iter(),
            self.state.get_termination_reason(),
            self.op,
            self.state.get_time(),
        ))
    }

    pub fn run_fast(mut self) -> Result<ArgminResult<O>, Error> {
        let total_time = std::time::Instant::now();

        let mut op_wrapper = OpWrapper::new(&self.op);
        let init_data = self.solver.init(&mut op_wrapper, &self.state)?;

        // If init() returned something, deal with it
        if let Some(data) = init_data {
            self.update(&data)?;
        }

        self.state.increment_func_counts(&op_wrapper);

        loop {
            // let state = self.to_state();

            // check first if it has already terminated
            // This should probably be solved better.
            // First, check if it isn't already terminated. If it isn't, evaluate the
            // stopping criteria. If `self.terminate()` is called without the checking
            // whether it has terminated already, then it may overwrite a termination set
            // within `next_iter()`!
            if !self.state.terminated() {
                self.state
                    .termination_reason(self.solver.terminate_internal(&self.state));
            }
            // Now check once more if the algorithm has terminated. If yes, then break.
            if self.state.terminated() {
                break;
            }

            let data = self.solver.next_iter(&mut op_wrapper, &self.state)?;

            self.state.increment_func_counts(&op_wrapper);

            self.update(&data)?;

            // increment iteration number
            self.state.increment_iter();

            self.checkpoint.store_cond(&self, self.state.get_iter())?;

            self.state.time(total_time.elapsed());

            // Check if termination occured inside next_iter()
            if self.state.terminated() {
                break;
            }
        }

        Ok(ArgminResult::new(
            self.state.get_best_param(),
            self.state.get_best_cost(),
            self.state.get_iter(),
            self.state.get_termination_reason(),
            self.op,
            self.state.get_time(),
        ))
    }

    /// Attaches a observer which implements `ArgminLog` to the solver.
    pub fn add_observer<OBS: Observe<O> + 'static>(
        mut self,
        observer: OBS,
        mode: ObserverMode,
    ) -> Self {
        self.observers.push(observer, mode);
        self
    }

    pub fn max_iters(mut self, iters: u64) -> Self {
        self.state.max_iters(iters);
        self
    }

    pub fn target_cost(mut self, cost: f64) -> Self {
        self.state.target_cost(cost);
        self
    }

    pub fn cost(mut self, cost: f64) -> Self {
        self.state.cost(cost);
        self
    }

    pub fn grad(mut self, grad: O::Param) -> Self {
        self.state.grad(grad);
        self
    }

    pub fn hessian(mut self, hessian: O::Hessian) -> Self {
        self.state.hessian(hessian);
        self
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
