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

// #![feature(specialization)]
#![feature(doc_spotlight)]
// necessary to get clippy to shut up about the s! macro
// remove this as soon as modified cholesky stuff is removed.
#![allow(clippy::deref_addrof)]
// I really do not like the a..=b syntax
#![allow(clippy::range_plus_one)]

#[cfg(feature = "ctrlc")]
pub extern crate ctrlc;
pub extern crate failure;
#[macro_use]
pub extern crate failure_derive;
#[macro_use]
extern crate slog;
#[cfg(feature = "ndarrayl")]
extern crate ndarray;
#[cfg(feature = "ndarrayl")]
extern crate ndarray_linalg;
extern crate rand;
extern crate slog_async;
extern crate slog_json;
extern crate slog_term;

/// Macros
#[macro_use]
pub mod macros;
/// base struct
mod base;
/// Error handling
mod errors;
/// Key value datastructure
mod kv;
/// Logging
mod logging;
/// Math utilities
mod math;
/// Output
mod output;
/// Definition of the return type of the solvers
mod result;
/// Definition of termination reasons
mod termination;

// TODO: Maybe leave logging/output stuff in its namespace
pub use crate::base::ArgminBase;
pub use crate::errors::*;
pub use crate::kv::ArgminKV;
pub use crate::logging::slog_logger::ArgminSlogLogger;
pub use crate::logging::ArgminLogger;
pub use crate::math::*;
pub use crate::output::file::WriteToFile;
pub use crate::output::ArgminWriter;
pub use crate::result::ArgminResult;
pub use crate::termination::TerminationReason;
pub use failure::Error;

/// Defines the interface to a solver. Usually, there is no need to implement this manually, use
/// the `argmin_derive` crate instead.
#[doc(spotlight)]
pub trait ArgminSolver: ArgminNextIter {
    /// apply cost function or operator to a parameter vector
    fn apply(
        &mut self,
        param: &<Self as ArgminNextIter>::Parameters,
    ) -> Result<<Self as ArgminNextIter>::OperatorOutput, Error>;

    /// compute the gradient for a parameter vector
    fn gradient(
        &mut self,
        param: &<Self as ArgminNextIter>::Parameters,
    ) -> Result<<Self as ArgminNextIter>::Parameters, Error>;

    /// compute the hessian for a parameter vector
    fn hessian(
        &mut self,
        param: &<Self as ArgminNextIter>::Parameters,
    ) -> Result<<Self as ArgminNextIter>::Hessian, Error>;

    /// modify the parameter vector
    fn modify(
        &self,
        param: &<Self as ArgminNextIter>::Parameters,
        extent: f64,
    ) -> Result<<Self as ArgminNextIter>::Parameters, Error>;

    /// return current parameter vector
    fn cur_param(&self) -> <Self as ArgminNextIter>::Parameters;

    /// return current gradient
    fn cur_grad(&self) -> <Self as ArgminNextIter>::Parameters;

    /// return current gradient
    fn cur_hessian(&self) -> <Self as ArgminNextIter>::Hessian;

    /// set current parameter vector
    fn set_cur_param(&mut self, param: <Self as ArgminNextIter>::Parameters);

    /// set current gradient
    fn set_cur_grad(&mut self, grad: <Self as ArgminNextIter>::Parameters);

    /// set current gradient
    fn set_cur_hessian(&mut self, hessian: <Self as ArgminNextIter>::Hessian);

    /// set current parameter vector
    fn set_best_param(&mut self, param: <Self as ArgminNextIter>::Parameters);

    /// Execute the optimization algorithm.
    fn run(&mut self) -> Result<ArgminResult<<Self as ArgminNextIter>::Parameters>, Error>;

    /// Execute the optimization algorithm without Ctrl-C handling, logging, writing and anything
    /// else which may cost unnecessary time.
    fn run_fast(&mut self) -> Result<ArgminResult<<Self as ArgminNextIter>::Parameters>, Error>;

    /// Returns the best solution found during optimization.
    fn result(&self) -> ArgminResult<<Self as ArgminNextIter>::Parameters>;

    /// Set termination reason (doesn't terminate yet! -- this is helpful for terminating within
    /// the iterations)
    fn set_termination_reason(&mut self, reason: TerminationReason);

    /// Evaluate all stopping criterions and return the `TerminationReason`
    fn terminate(&mut self) -> TerminationReason;

    /// Set max number of iterations.
    fn set_max_iters(&mut self, iters: u64);

    /// Get max number of iterations.
    fn max_iters(&self) -> u64;

    /// Get current iteration number.
    fn cur_iter(&self) -> u64;

    /// Increment the iteration number by one
    fn increment_iter(&mut self);

    /// Get current cost function value
    fn cur_cost(&self) -> f64;

    /// Get current cost function value
    fn set_cur_cost(&mut self, cost: f64);

    /// Get best cost function value
    // fn best_cost(&self) -> <Self as ArgminNextIter>::OperatorOutput;
    fn best_cost(&self) -> f64;

    /// set best cost value
    // fn set_best_cost(&mut self, <Self as ArgminNextIter>::OperatorOutput);
    fn set_best_cost(&mut self, cost: f64);

    /// Set the target cost function value which is used as a stopping criterion
    fn set_target_cost(&mut self, cost: f64);

    /// Add a logger to the array of loggers
    fn add_logger(&mut self, logger: Box<ArgminLog>);

    /// Add a writer to the array of writers
    fn add_writer(&mut self, writer: Box<ArgminWrite<Param = Self::Parameters>>);

    /// Reset the base of the algorithm to its initial state
    fn base_reset(&mut self);

    /// Increment the cost function evaluation count
    fn increment_cost_func_count(&mut self);

    /// Increaese the cost function evaluation count by a given value
    fn increase_cost_func_count(&mut self, count: u64);

    /// Return the cost function evaluation count
    fn cost_func_count(&self) -> u64;

    /// Increment the gradient evaluation count
    fn increment_grad_func_count(&mut self);

    /// Increase the gradient evaluation count by a given value
    fn increase_grad_func_count(&mut self, count: u64);

    /// Return the gradient evaluation count
    fn grad_func_count(&self) -> u64;

    /// Increment the hessian evaluation count
    fn increment_hessian_func_count(&mut self);

    /// Increase the hessian evaluation count by a given value
    fn increase_hessian_func_count(&mut self, count: u64);

    /// Return the gradient evaluation count
    fn hessian_func_count(&self) -> u64;
}

/// Main part of every solver: `next_iter` computes one iteration of the algorithm and `init` is
/// executed before these iterations. The `init` method comes with a default implementation which
/// corresponds to doing nothing.
pub trait ArgminNextIter {
    /// Parameter vectors
    type Parameters: Clone;
    /// Output of the operator
    type OperatorOutput;
    /// Hessian
    type Hessian;

    /// Computes one iteration of the algorithm.
    fn next_iter(&mut self) -> Result<ArgminIterationData<Self::Parameters>, Error>;

    /// Initializes the algorithm
    ///
    /// This is executed before any iterations are performed. It can be used to perform
    /// precomputations. The default implementation corresponds to doing nothing.
    fn init(&mut self) -> Result<(), Error> {
        Ok(())
    }
}

/// Defince the interface every logger needs to expose
pub trait ArgminLog {
    /// Logs general information (a message `msg` and/or key-value pairs `kv`).
    fn log_info(&self, msg: &str, kv: &ArgminKV) -> Result<(), Error>;

    /// Logs information from iterations. Only accepts key-value pairs. `log_iter` is made to log
    /// to a database or a CSV file. Therefore the structure of the key-value pairs should not
    /// change inbetween iterations.
    fn log_iter(&self, kv: &ArgminKV) -> Result<(), Error>;
}

/// Every writer (which is something that writes parameter vectors somewhere after each iteration)
/// needs to implement this.
pub trait ArgminWrite {
    type Param;

    /// Writes the parameter vector somewhere
    fn write(&self, param: &Self::Param) -> Result<(), Error>;
}

/// The datastructure which is returned by the `next_iter` method of the `ArgminNextIter` trait.
///
/// TODO: think about removing this or replacing it with something better. Actually, a tuple would
/// be sufficient.
pub struct ArgminIterationData<T: Clone> {
    /// Current parameter vector
    param: T,
    /// Current cost function value
    cost: f64,
    /// Key value pairs which are currently only used to provide additional information for the
    /// loggers
    kv: Option<ArgminKV>,
}

impl<T: Clone> ArgminIterationData<T> {
    /// Constructor
    pub fn new(param: T, cost: f64) -> Self {
        ArgminIterationData {
            param,
            cost,
            kv: None,
        }
    }

    /// Returns the parameter vector
    pub fn param(&self) -> T {
        self.param.clone()
    }

    /// Returns the cost function value
    pub fn cost(&self) -> f64 {
        self.cost
    }

    /// Adds an `ArgminKV`
    pub fn add_kv(&mut self, kv: ArgminKV) -> &mut Self {
        self.kv = Some(kv);
        self
    }

    /// Returns an `ArgminKV`
    ///
    /// TODO: Keep it consistent, remove the `get_`.
    pub fn get_kv(&self) -> Option<ArgminKV> {
        self.kv.clone()
    }
}

/// This trait needs to be implemented for every operator/cost function.
///
/// It is required to implement the `apply` method, all others are optional and provide a default
/// implementation which is essentially returning an error which indicates that the method has not
/// been implemented. Those methods (`gradient` and `modify`) only need to be implemented if the
/// uses solver requires it.
pub trait ArgminOperator {
    /// Type of the parameter vector
    type Parameters;
    /// Output of the operator. Most solvers expect `f64`.
    type OperatorOutput;
    /// Type of Hessian
    type Hessian;

    /// Applies the operator/cost function to parameters
    fn apply(&self, _param: &Self::Parameters) -> Result<Self::OperatorOutput, Error> {
        Err(ArgminError::NotImplemented {
            text: "Method `apply` of ArgminOperator trait not implemented!".to_string(),
        }
        .into())
    }

    /// Computes the gradient at the given parameters
    fn gradient(&self, _param: &Self::Parameters) -> Result<Self::Parameters, Error> {
        Err(ArgminError::NotImplemented {
            text: "Method `gradient` of ArgminOperator trait not implemented!".to_string(),
        }
        .into())
    }

    /// Computes the hessian at the given parameters
    fn hessian(&self, _param: &Self::Parameters) -> Result<Self::Hessian, Error> {
        Err(ArgminError::NotImplemented {
            text: "Method `hessian` of ArgminOperator trait not implemented!".to_string(),
        }
        .into())
    }

    /// Modifies a parameter vector. Comes with a variable that indicates the "degree" of the
    /// modification.
    fn modify(&self, _param: &Self::Parameters, _extent: f64) -> Result<Self::Parameters, Error> {
        Err(ArgminError::NotImplemented {
            text: "Method `modify` of ArgminOperator trait not implemented!".to_string(),
        }
        .into())
    }
}

#[derive(Clone, Default)]
pub struct NoOperator<T: Clone, U: Clone, H: Clone> {
    param: std::marker::PhantomData<*const T>,
    output: std::marker::PhantomData<*const U>,
    hessian: std::marker::PhantomData<*const H>,
}

impl<T: Clone, U: Clone, H: Clone> NoOperator<T, U, H> {
    pub fn new() -> Self {
        NoOperator {
            param: std::marker::PhantomData,
            output: std::marker::PhantomData,
            hessian: std::marker::PhantomData,
        }
    }
}

impl<T, U, H> ArgminOperator for NoOperator<T, U, H>
where
    T: Clone + std::default::Default,
    U: Clone + std::default::Default,
    H: Clone + std::default::Default,
{
    type Parameters = T;
    type OperatorOutput = U;
    type Hessian = H;

    fn apply(&self, _p: &Self::Parameters) -> Result<Self::OperatorOutput, Error> {
        Ok(Self::OperatorOutput::default())
    }

    fn gradient(&self, _p: &Self::Parameters) -> Result<Self::Parameters, Error> {
        Ok(Self::Parameters::default())
    }

    fn hessian(&self, _p: &Self::Parameters) -> Result<Self::Hessian, Error> {
        Ok(Self::Hessian::default())
    }

    fn modify(&self, _p: &Self::Parameters, _t: f64) -> Result<Self::Parameters, Error> {
        Ok(Self::Parameters::default())
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
pub trait ArgminLineSearch: ArgminSolver {
    /// Set the initial parameter (starting point)
    fn set_initial_parameter(&mut self, param: <Self as ArgminNextIter>::Parameters);

    /// Set the search direction
    fn set_search_direction(&mut self, direction: <Self as ArgminNextIter>::Parameters);

    /// Set the initial step length
    fn set_initial_alpha(&mut self, step_length: f64) -> Result<(), Error>;

    /// Set the cost function value at the starting point as opposed to computing it (see
    /// `calc_initial_cost`)
    fn set_initial_cost(&mut self, cost: f64);

    /// Set the gradient at the starting point as opposed to computing it (see
    /// `calc_initial_gradient`)
    fn set_initial_gradient(&mut self, grad: <Self as ArgminNextIter>::Parameters);

    /// calculate the initial cost function value using an operator as opposed to setting it
    /// manually (see `set_initial_cost`)
    fn calc_initial_cost(&mut self) -> Result<(), Error>;

    /// calculate the initial gradient using an operator as opposed to setting it manually (see
    /// `set_initial_gradient`)
    fn calc_initial_gradient(&mut self) -> Result<(), Error>;
}

/// Defines a common interface to methods which calculate approximate steps for trust region
/// methods. Requires that `ArgminSolver` is implemented as well.
pub trait ArgminTrustRegion: ArgminSolver {
    // /// Set the initial parameter (starting point)
    // fn set_initial_parameter(&mut self, <Self as ArgminNextIter>::Parameters);

    /// Set the initial step length
    fn set_radius(&mut self, radius: f64);

    /// Set the gradient at the starting point
    fn set_grad(&mut self, grad: <Self as ArgminNextIter>::Parameters);

    /// Set the gradient at the starting point
    fn set_hessian(&mut self, hessian: <Self as ArgminNextIter>::Hessian);
}

/// Every method for the update of beta needs to implement this trait.
pub trait ArgminNLCGBetaUpdate<T> {
    /// Update beta
    /// Parameter 1: \nabla f_k
    /// Parameter 2: \nabla f_{k+1}
    /// Parameter 3: p_k
    fn update(&self, nabla_f_k: &T, nabla_f_k_p_1: &T, p_k: &T) -> f64;
}
