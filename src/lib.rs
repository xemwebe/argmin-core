// Copyright 2018-2020-2020 argmin developers
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
/// Math utilities
mod math;
/// Phony Operator
// #[cfg(test)]
mod nooperator;
/// Observers;
mod observers;
/// Wrapper around operators which keeps track of function evaluation counts
mod opwrapper;
/// Definition of the return type of the solvers
mod result;
/// Serialization of `ArgminSolver`s
#[cfg(feature = "serde1")]
mod serialization;
/// Definition of termination reasons
mod termination;

pub use crate::errors::*;
pub use crate::executor::*;
pub use crate::iterstate::*;
pub use crate::kv::ArgminKV;
pub use crate::math::*;
pub use crate::nooperator::*;
pub use crate::observers::*;
pub use crate::opwrapper::*;
pub use crate::result::ArgminResult;
pub use crate::termination::TerminationReason;
pub use failure::Error;
#[cfg(feature = "serde1")]
use serde::de::DeserializeOwned;
#[cfg(feature = "serde1")]
use serde::Serialize;
#[cfg(feature = "serde1")]
pub use serialization::*;


// serde feature is on
#[cfg(feature = "serde1")]
pub trait SerializeAlias: Serialize {}
#[cfg(feature = "serde1")]
impl<T> SerializeAlias for T where T: Serialize {}
#[cfg(feature = "serde1")]
pub trait DeserializeOwnedAlias: DeserializeOwned {}
#[cfg(feature = "serde1")]
impl<T> DeserializeOwnedAlias for T where T: DeserializeOwned {}

// serde feature is off
#[cfg(not(feature = "serde1"))]
pub trait SerializeAlias {}
#[cfg(not(feature = "serde1"))]
impl<T> SerializeAlias for T {}
#[cfg(not(feature = "serde1"))]
pub trait DeserializeOwnedAlias {}
#[cfg(not(feature = "serde1"))]
impl<T> DeserializeOwnedAlias for T {}

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
pub trait ArgminOp: Clone + Send + Sync + SerializeAlias {
    // TODO: Once associated type defaults are stable, it hopefully will be possible to define
    // default types for `Hessian` and `Jacobian`.
    /// Type of the parameter vector
    type Param: Clone + SerializeAlias + DeserializeOwnedAlias;
    /// Output of the operator
    type Output: Clone + SerializeAlias + DeserializeOwnedAlias;
    /// Type of Hessian
    type Hessian: Clone + SerializeAlias + DeserializeOwnedAlias;
    /// Type of Jacobian
    type Jacobian: Clone + SerializeAlias + DeserializeOwnedAlias;

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

    /// Computes the Hessian at the given parameters
    fn hessian(&self, _param: &Self::Param) -> Result<Self::Hessian, Error> {
        Err(ArgminError::NotImplemented {
            text: "Method `hessian` of ArgminOp trait not implemented!".to_string(),
        }
        .into())
    }

    /// Computes the Hessian at the given parameters
    fn jacobian(&self, _param: &Self::Param) -> Result<Self::Jacobian, Error> {
        Err(ArgminError::NotImplemented {
            text: "Method `jacobian` of ArgminOp trait not implemented!".to_string(),
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


pub trait Solver<O: ArgminOp>: SerializeAlias {
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

    /// Checks whether basic termination reasons apply.
    ///
    /// Terminate if
    ///
    /// 1) algorithm was terminated somewhere else in the Executor
    /// 2) iteration count exceeds maximum number of iterations
    /// 3) cost is lower than target cost
    ///
    /// This can be overwritten in a `Solver` implementation; however it is not advised.
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

    /// Checks whether the algorithm must be terminated
    fn terminate(&mut self, _state: &IterState<O>) -> TerminationReason {
        TerminationReason::NotTerminated
    }
}

/// The datastructure which is returned by the `next_iter` method of the `Solver` trait.
///
/// TODO: Rename to IterResult?
#[cfg_attr(feature = "serde1", derive(Serialize))]
#[derive(Clone, Debug, Default)]
pub struct ArgminIterData<O: ArgminOp> {
    /// Current parameter vector
    param: Option<O::Param>,
    /// Current cost function value
    cost: Option<f64>,
    /// Current gradient
    grad: Option<O::Param>,
    /// Current Hessian
    hessian: Option<O::Hessian>,
    /// Current Jacobian
    jacobian: Option<O::Jacobian>,
    population: Option<Vec<(O::Param, f64)>>,
    /// termination reason
    termination_reason: Option<TerminationReason>,
    /// Key value pairs which are used to provide additional information for the Observers
    kv: ArgminKV,
}

// TODO: Many clones are necessary in the getters.. maybe a complete "deconstruct" method would be
// better?
impl<O: ArgminOp> ArgminIterData<O> {
    /// Constructor
    pub fn new() -> Self {
        ArgminIterData {
            param: None,
            cost: None,
            grad: None,
            hessian: None,
            jacobian: None,
            termination_reason: None,
            population: None,
            kv: make_kv!(),
        }
    }

    /// Set parameter vector
    pub fn param(mut self, param: O::Param) -> Self {
        self.param = Some(param);
        self
    }

    /// Set cost function value
    pub fn cost(mut self, cost: f64) -> Self {
        self.cost = Some(cost);
        self
    }

    /// Set gradient
    pub fn grad(mut self, grad: O::Param) -> Self {
        self.grad = Some(grad);
        self
    }

    /// Set Hessian
    pub fn hessian(mut self, hessian: O::Hessian) -> Self {
        self.hessian = Some(hessian);
        self
    }

    /// Set Jacobian
    pub fn jacobian(mut self, jacobian: O::Jacobian) -> Self {
        self.jacobian = Some(jacobian);
        self
    }

    /// Set Population
    pub fn population(mut self, population: Vec<(O::Param, f64)>) -> Self {
        self.population = Some(population);
        self
    }

    /// Adds an `ArgminKV`
    pub fn kv(mut self, kv: ArgminKV) -> Self {
        self.kv = kv;
        self
    }

    /// Set termination reason
    pub fn termination_reason(mut self, reason: TerminationReason) -> Self {
        self.termination_reason = Some(reason);
        self
    }

    /// Get parameter vector
    pub fn get_param(&self) -> Option<O::Param> {
        self.param.clone()
    }

    /// Get cost function value
    pub fn get_cost(&self) -> Option<f64> {
        self.cost
    }

    /// Get gradient
    pub fn get_grad(&self) -> Option<O::Param> {
        self.grad.clone()
    }

    /// Get Hessian
    pub fn get_hessian(&self) -> Option<O::Hessian> {
        self.hessian.clone()
    }

    /// Get Jacobian
    pub fn get_jacobian(&self) -> Option<O::Jacobian> {
        self.jacobian.clone()
    }

    /// Get reference to population
    pub fn get_population(&self) -> Option<&Vec<(O::Param, f64)>> {
        match &self.population {
            Some(population) => Some(&population),
            None => None,
        }
    }

    /// Get termination reason
    pub fn get_termination_reason(&self) -> Option<TerminationReason> {
        self.termination_reason
    }

    /// Return KV
    pub fn get_kv(&self) -> ArgminKV {
        self.kv.clone()
    }
}

/// Defines a common interface for line search methods.
pub trait ArgminLineSearch<P>: SerializeAlias {
    /// Set the search direction
    fn set_search_direction(&mut self, direction: P);

    /// Set the initial step length
    fn set_init_alpha(&mut self, step_length: f64) -> Result<(), Error>;
}

/// Defines a common interface to methods which calculate approximate steps for trust region
/// methods.
pub trait ArgminTrustRegion: Clone + SerializeAlias {
    /// Set the initial step length
    fn set_radius(&mut self, radius: f64);
}

/// Common interface for beta update methods (Nonlinear-CG)
pub trait ArgminNLCGBetaUpdate<T>: SerializeAlias {
    /// Update beta
    /// Parameter 1: \nabla f_k
    /// Parameter 2: \nabla f_{k+1}
    /// Parameter 3: p_k
    fn update(&self, nabla_f_k: &T, nabla_f_k_p_1: &T, p_k: &T) -> f64;
}
