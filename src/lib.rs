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

/// Macros
#[macro_use]
pub mod macros;
/// Error handling
mod errors;
/// Executor
pub mod executor;
/// iteration state
mod iterstate;
/// Key value datastructure
mod kv;
/// Logging
mod logging;
/// Math utilities
mod math;
/// Phony Operator
// #[cfg(test)]
mod nooperator;
/// Wrapper around operators which keeps track of function evaluation counts
mod opwrapper;
/// Output
mod output;
/// Definition of the return type of the solvers
mod result;
/// Serialization of `ArgminSolver`s
mod serialization;
/// Definition of termination reasons
mod termination;

// TODO: Maybe leave logging/output stuff in its namespace
pub use crate::errors::*;
pub use crate::executor::*;
pub use crate::iterstate::*;
pub use crate::kv::ArgminKV;
pub use crate::logging::slog_logger::ArgminSlogLogger;
pub use crate::logging::Observer;
pub use crate::math::*;
pub use crate::nooperator::*;
pub use crate::opwrapper::*;
pub use crate::output::file::{WriteToFile, WriteToFileSerializer};
pub use crate::output::*;
pub use crate::result::ArgminResult;
pub use crate::termination::TerminationReason;
pub use failure::Error;
use serde::de::DeserializeOwned;
use serde::Serialize;
pub use serialization::*;

pub mod finitediff {
    //! Finite Differentiation
    //!
    //! Reexport of `finitediff` crate.
    pub use finitediff::*;
}

/// This trait needs to be implemented for every operator/cost function.
///
/// It is required to implement the `apply` method, all others are optional and provide a default
/// implementation which is essentially returning an error which indicates that the method has not
/// been implemented. Those methods (`gradient` and `modify`) only need to be implemented if the
/// uses solver requires it.
pub trait ArgminOp: Clone + Send + Sync + Serialize {
    /// Type of the parameter vector
    type Param: Clone + Serialize + DeserializeOwned;
    /// Output of the operator
    type Output;
    /// Type of Hessian
    type Hessian: Clone + Serialize + DeserializeOwned;

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

pub trait Solver<O: ArgminOp>: Serialize {
    const NAME: &'static str = "UNDEFINED";

    /// Computes one iteration of the algorithm.
    fn next_iter(
        &mut self,
        op: &mut OpWrapper<O>,
        state: &IterState<O>,
    ) -> Result<ArgminIterData<O>, Error>;

    /// Initializes the algorithm
    ///
    /// This is executed before any iterations are performed. It can be used to perform
    /// precomputations. The default implementation corresponds to doing nothing.
    fn init(
        &mut self,
        _op: &mut OpWrapper<O>,
        _state: &IterState<O>,
    ) -> Result<Option<ArgminIterData<O>>, Error> {
        Ok(None)
    }

    fn terminate_internal(&mut self, state: &IterState<O>) -> TerminationReason {
        let solver_terminate = self.terminate(state);
        if solver_terminate.terminated() {
            return solver_terminate;
        }
        if state.get_iter() >= state.get_max_iters() {
            return TerminationReason::MaxItersReached;
        }
        if state.get_cost() <= state.get_target_cost() {
            return TerminationReason::TargetCostReached;
        }
        TerminationReason::NotTerminated
    }

    fn terminate(&mut self, _state: &IterState<O>) -> TerminationReason {
        TerminationReason::NotTerminated
    }
}

/// Defince the interface every Observer needs to expose
pub trait Observe<O: ArgminOp>: Send + Sync {
    fn observe_init(&self, msg: &str, kv: &ArgminKV) -> Result<(), Error>;

    fn observe_iter(&self, state: &IterState<O>, kv: &ArgminKV) -> Result<(), Error>;
}

/// Every writer (which is something that writes parameter vectors somewhere after each iteration)
/// needs to implement this.
pub trait ArgminWrite: Send + Sync {
    type Param;

    /// Writes the parameter vector somewhere
    fn write(&self, param: &Self::Param, iter: u64, new_best: bool) -> Result<(), Error>;
}

/// The datastructure which is returned by the `next_iter` method of the `Solver` trait.
#[derive(Clone, Serialize, Debug, Default)]
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
    /// Observers
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
pub trait ArgminLineSearch<P>: Serialize {
    /// Set the search direction
    fn set_search_direction(&mut self, direction: P);

    /// Set the initial step length
    fn set_init_alpha(&mut self, step_length: f64) -> Result<(), Error>;
}

/// Defines a common interface to methods which calculate approximate steps for trust region
/// methods. Requires that `ArgminSolver` is implemented as well.
pub trait ArgminTrustRegion: Clone + Serialize {
    /// Set the initial step length
    fn set_radius(&mut self, radius: f64);
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
