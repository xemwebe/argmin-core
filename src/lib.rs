// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Argmin Optimizaton toolbox core
//!
//! This crate contains the core functionality of argmin. If you just want to run an optimization
//! method, this is *not* what you are looking for. However, if you want to implement your own
//! solver based on the argmin architecture, you should find all necessary tools here.
//!
//! TODOs:
//!   * Provide an example of how to implement a solver

// I really do not like the a..=b syntax
#![allow(clippy::range_plus_one)]

#[cfg(feature = "ctrlc")]
pub extern crate ctrlc;

/// Macros
#[macro_use]
pub mod macros;
// /// base struct
// mod base;
/// Error handling
mod errors;
/// Executor
pub mod executor;
/// Key value datastructure
mod kv;
/// Logging
mod logging;
/// Math utilities
mod math;
/// Phony Operator
// #[cfg(test)]
mod nooperator;
/// Output
mod output;
/// Definition of the return type of the solvers
mod result;
/// Serialization of `ArgminSolver`s
mod serialization;
/// Definition of termination reasons
mod termination;

// TODO: Maybe leave logging/output stuff in its namespace
// pub use crate::base::ArgminBase;
pub use crate::errors::*;
pub use crate::executor::*;
pub use crate::kv::ArgminKV;
pub use crate::logging::slog_logger::ArgminSlogLogger;
pub use crate::logging::ArgminLogger;
pub use crate::math::*;
pub use crate::nooperator::*;
pub use crate::output::file::{WriteToFile, WriteToFileSerializer};
pub use crate::output::*;
pub use crate::result::ArgminResult;
pub use crate::termination::TerminationReason;
pub use failure::Error;
// use serde::de::DeserializeOwned;
use serde::Serialize;
pub use serialization::*;
// use std::path::Path;

pub mod finitediff {
    //! Finite Differentiation
    //!
    //! Reexport of `finitediff` crate.
    pub use finitediff::*;
}

/// Defines the interface to a solver. Usually, there is no need to implement this manually, use
/// the `argmin_derive` crate instead.
// pub trait ArgminSolver: ArgminIter {
//     /// Load solver from checkpoint
//     fn from_checkpoint<P: AsRef<Path>>(path: P) -> Result<Self, Error>
//     where
//         Self: Sized + DeserializeOwned,
//     {
//         load_checkpoint(path)
//     }
//
//     /// apply cost function or operator to a parameter vector
//     fn apply(
//         &mut self,
//         param: &<Self as ArgminIter>::Param,
//     ) -> Result<<Self as ArgminIter>::Output, Error>;
//
//     /// compute the gradient for a parameter vector
//     fn gradient(
//         &mut self,
//         param: &<Self as ArgminIter>::Param,
//     ) -> Result<<Self as ArgminIter>::Param, Error>;
//
//     /// compute the hessian for a parameter vector
//     fn hessian(
//         &mut self,
//         param: &<Self as ArgminIter>::Param,
//     ) -> Result<<Self as ArgminIter>::Hessian, Error>;
//
//     /// modify the parameter vector
//     fn modify(
//         &self,
//         param: &<Self as ArgminIter>::Param,
//         extent: f64,
//     ) -> Result<<Self as ArgminIter>::Param, Error>;
//
//     /// return current parameter vector
//     fn cur_param(&self) -> <Self as ArgminIter>::Param;
//
//     /// return current gradient
//     fn cur_grad(&self) -> <Self as ArgminIter>::Param;
//
//     /// return current gradient
//     fn cur_hessian(&self) -> <Self as ArgminIter>::Hessian;
//
//     /// set current parameter vector
//     fn set_cur_param(&mut self, param: <Self as ArgminIter>::Param);
//
//     /// set current gradient
//     fn set_cur_grad(&mut self, grad: <Self as ArgminIter>::Param);
//
//     /// set current gradient
//     fn set_cur_hessian(&mut self, hessian: <Self as ArgminIter>::Hessian);
//
//     /// set current parameter vector
//     fn set_best_param(&mut self, param: <Self as ArgminIter>::Param);
//
//     /// Execute the optimization algorithm.
//     fn run(&mut self) -> Result<ArgminResult<<Self as ArgminIter>::Param>, Error>;
//
//     /// Execute the optimization algorithm without Ctrl-C handling, logging, writing and anything
//     /// else which may cost unnecessary time.
//     fn run_fast(&mut self) -> Result<ArgminResult<<Self as ArgminIter>::Param>, Error>;
//
//     /// Returns the best solution found during optimization.
//     fn result(&self) -> ArgminResult<<Self as ArgminIter>::Param>;
//
//     /// Set termination reason (doesn't terminate yet! -- this is helpful for terminating within
//     /// the iterations)
//     fn set_termination_reason(&mut self, reason: TerminationReason);
//
//     /// Evaluate all stopping criterions and return the `TerminationReason`
//     fn terminate(&mut self) -> TerminationReason;
//
//     /// Set max number of iterations.
//     fn set_max_iters(&mut self, iters: u64);
//
//     /// Get max number of iterations.
//     fn max_iters(&self) -> u64;
//
//     /// Get current iteration number.
//     fn cur_iter(&self) -> u64;
//
//     /// Increment the iteration number by one
//     fn increment_iter(&mut self);
//
//     /// Get current cost function value
//     fn cur_cost(&self) -> f64;
//
//     /// Get previous cost function value
//     fn prev_cost(&self) -> f64;
//
//     /// Get current cost function value
//     fn set_cur_cost(&mut self, cost: f64);
//
//     /// Get best cost function value
//     // fn best_cost(&self) -> <Self as ArgminIter>::Output;
//     fn best_cost(&self) -> f64;
//
//     /// set best cost value
//     // fn set_best_cost(&mut self, <Self as ArgminIter>::Output);
//     fn set_best_cost(&mut self, cost: f64);
//
//     /// Set the target cost function value which is used as a stopping criterion
//     fn set_target_cost(&mut self, cost: f64);
//
//     /// Add a logger to the array of loggers
//     fn add_logger(&mut self, logger: std::sync::Arc<ArgminLog>);
//
//     /// Add a writer to the array of writers
//     fn add_writer(&mut self, writer: std::sync::Arc<ArgminWrite<Param = Self::Param>>);
//
//     /// Reset the base of the algorithm to its initial state
//     fn base_reset(&mut self);
//
//     /// Increment the cost function evaluation count
//     fn increment_cost_func_count(&mut self);
//
//     /// Increaese the cost function evaluation count by a given value
//     fn increase_cost_func_count(&mut self, count: u64);
//
//     /// Return the cost function evaluation count
//     fn cost_func_count(&self) -> u64;
//
//     /// Increment the gradient evaluation count
//     fn increment_grad_func_count(&mut self);
//
//     /// Increase the gradient evaluation count by a given value
//     fn increase_grad_func_count(&mut self, count: u64);
//
//     /// Return the gradient evaluation count
//     fn grad_func_count(&self) -> u64;
//
//     /// Increment the hessian evaluation count
//     fn increment_hessian_func_count(&mut self);
//
//     /// Increase the hessian evaluation count by a given value
//     fn increase_hessian_func_count(&mut self, count: u64);
//
//     /// Return the gradient evaluation count
//     fn hessian_func_count(&self) -> u64;
//
//     /// Set checkpoint directory
//     fn set_checkpoint_dir(&mut self, dir: &str);
//
//     /// Set checkpoint name
//     fn set_checkpoint_name(&mut self, dir: &str);
//
//     /// Set checkpoint mode
//     fn set_checkpoint_mode(&mut self, mode: CheckpointMode);
// }

/// Defince the interface every logger needs to expose
pub trait ArgminLog: Send + Sync {
    /// Logs general information (a message `msg` and/or key-value pairs `kv`).
    fn log_info(&self, msg: &str, kv: &ArgminKV) -> Result<(), Error>;

    /// Logs information from iterations. Only accepts key-value pairs. `log_iter` is made to log
    /// to a database or a CSV file. Therefore the structure of the key-value pairs should not
    /// change inbetween iterations.
    fn log_iter(&self, kv: &ArgminKV) -> Result<(), Error>;
}

/// Every writer (which is something that writes parameter vectors somewhere after each iteration)
/// needs to implement this.
pub trait ArgminWrite: Send + Sync {
    type Param;

    /// Writes the parameter vector somewhere
    fn write(&self, param: &Self::Param, iter: u64, new_best: bool) -> Result<(), Error>;
}

/// The datastructure which is returned by the `next_iter` method of the `ArgminIter` trait.
///
/// TODO: think about removing this or replacing it with something better. Actually, a tuple would
/// be sufficient.
#[derive(Clone, Serialize, Debug)]
pub struct ArgminIterData<O: ArgminOp> {
    /// Current parameter vector
    param: Option<O::Param>,
    /// Current cost function value
    cost: Option<f64>,
    /// Current gradient
    grad: Option<O::Param>,
    /// Current gradient
    hessian: Option<O::Hessian>,
    /// terminationreason
    termination_reason: Option<TerminationReason>,
    /// Key value pairs which are currently only used to provide additional information for the
    /// loggers
    kv: Option<ArgminKV>,
}

impl<O: ArgminOp> ArgminIterData<O> {
    /// Constructor
    pub fn new() -> Self {
        ArgminIterData {
            param: None,
            cost: None,
            grad: None,
            hessian: None,
            termination_reason: None,
            kv: None,
        }
    }

    pub fn param(mut self, param: O::Param) -> Self {
        self.param = Some(param);
        self
    }

    pub fn cost(mut self, cost: f64) -> Self {
        self.cost = Some(cost);
        self
    }

    pub fn grad(mut self, grad: O::Param) -> Self {
        self.grad = Some(grad);
        self
    }

    pub fn hessian(mut self, hessian: O::Hessian) -> Self {
        self.hessian = Some(hessian);
        self
    }

    /// Adds an `ArgminKV`
    pub fn kv(mut self, kv: ArgminKV) -> Self {
        self.kv = Some(kv);
        self
    }

    pub fn termination_reason(mut self, reason: TerminationReason) -> Self {
        self.termination_reason = Some(reason);
        self
    }

    pub fn get_param(&self) -> Option<O::Param> {
        self.param.clone()
    }

    pub fn get_cost(&self) -> Option<f64> {
        self.cost
    }

    pub fn get_grad(&self) -> Option<O::Param> {
        self.grad.clone()
    }

    pub fn get_hessian(&self) -> Option<O::Hessian> {
        self.hessian.clone()
    }

    pub fn get_termination_reason(&self) -> Option<TerminationReason> {
        self.termination_reason
    }

    /// Returns an `ArgminKV`
    pub fn get_kv(&self) -> Option<ArgminKV> {
        self.kv.clone()
    }
}

/// Defines a common interface to line search methods. Requires that `ArgminSolver` is implemented
/// for the line search method as well.
///
/// The cost function value and the gradient at the starting point can either be provided
/// (`set_initial_cost` and `set_initial_gradient`) or they can be computed using the operator from
/// the implementation of `ArgminSolver` (see `calc_initial_cost` and `calc_initial_gradient`). The
/// former is convenient if cost and gradient at the starting point are already known for some
/// reason (i.e. the solver which uses the line search has already computed cost and gradient) and
/// avoids unneccessary computation of those values.
// pub trait ArgminLineSearch<O: ArgminOp>: Solver<O> + Serialize {
pub trait ArgminLineSearch<P>: Serialize {
    /// Set the initial parameter (starting point)
    fn set_init_param(&mut self, param: P);

    /// Set the search direction
    fn set_search_direction(&mut self, direction: P);

    /// Set the initial step length
    fn set_init_alpha(&mut self, step_length: f64) -> Result<(), Error>;

    /// Set the cost function value at the starting point as opposed to computing it (see
    /// `calc_initial_cost`)
    fn set_init_cost(&mut self, cost: f64);

    /// Set the gradient at the starting point as opposed to computing it (see
    /// `calc_initial_gradient`)
    fn set_init_grad(&mut self, grad: P);
}

/// Defines a common interface to methods which calculate approximate steps for trust region
/// methods. Requires that `ArgminSolver` is implemented as well.
pub trait ArgminTrustRegion<P, H>: Clone + Serialize {
    /// Set the initial step length
    fn set_radius(&mut self, radius: f64);

    /// Set the gradient at the starting point
    fn set_grad(&mut self, grad: P);

    /// Set the gradient at the starting point
    fn set_hessian(&mut self, hessian: H);
}
//
/// Every method for the update of beta needs to implement this trait.
pub trait ArgminNLCGBetaUpdate<T>: Serialize {
    /// Update beta
    /// Parameter 1: \nabla f_k
    /// Parameter 2: \nabla f_{k+1}
    /// Parameter 3: p_k
    fn update(&self, nabla_f_k: &T, nabla_f_k_p_1: &T, p_k: &T) -> f64;
}
