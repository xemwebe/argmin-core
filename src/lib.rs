// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Argmin Optimizaton toolbox
//!
//! TODO: Documentation.

pub extern crate ctrlc;
pub extern crate failure;
#[macro_use]
pub extern crate failure_derive;
#[macro_use]
extern crate slog;
extern crate rand;
extern crate slog_async;
extern crate slog_json;
extern crate slog_term;
// pub use failure::format_err;

/// base struct
mod base;
/// Error handling
mod errors;
/// Key value datastructure
mod kv;
/// Logging
mod logging;
/// Macros
pub mod macros;
/// Math utilities
mod math;
/// Output
mod output;
/// Definition of the return type of the solvers
mod result;
/// Definition of termination reasons
mod termination;

// TODO: Maybe leave logging/output stuff in its namespace
pub use base::ArgminBase;
pub use errors::*;
pub use failure::Error;
pub use kv::ArgminKV;
pub use logging::slog_logger::ArgminSlogLogger;
pub use logging::ArgminLogger;
pub use math::*;
pub use output::file::WriteToFile;
pub use output::ArgminWriter;
pub use result::ArgminResult;
pub use termination::TerminationReason;

/// Every solver must implement this trait
pub trait ArgminSolver: ArgminNextIter {
    /// apply cost function or operator to parameter
    fn apply(
        &mut self,
        &<Self as ArgminNextIter>::Parameters,
    ) -> Result<<Self as ArgminNextIter>::OperatorOutput, Error>;

    /// compute gradient
    fn gradient(
        &mut self,
        &<Self as ArgminNextIter>::Parameters,
    ) -> Result<<Self as ArgminNextIter>::Parameters, Error>;

    /// modify parameter vector
    fn modify(
        &mut self,
        &<Self as ArgminNextIter>::Parameters,
        f64,
    ) -> Result<<Self as ArgminNextIter>::Parameters, Error>;

    /// Runs the algorithm. Created by the `make_run!` macro.
    fn run(&mut self) -> Result<ArgminResult<<Self as ArgminNextIter>::Parameters>, Error>;

    /// Returns the result.
    fn result(&self) -> ArgminResult<<Self as ArgminNextIter>::Parameters>;

    /// Evaluate all stopping criterions and return the `TerminationReason`
    /// Implemented by the `make_terminate!` macro.
    fn terminate(&mut self) -> TerminationReason;

    /// Set max number of iterations.
    /// I'd like to return `&mut Self` but then `ArgminSolver` cannot be turned into a trait object
    /// anymore... :/
    fn set_max_iters(&mut self, u64);

    fn set_target_cost(&mut self, f64);

    fn add_logger(&mut self, Box<ArgminLog>);
    fn add_writer(&mut self, Box<ArgminWrite<Param = Self::Parameters>>);

    fn reset(&mut self);
}

pub trait ArgminNextIter {
    type Parameters: Clone;
    type OperatorOutput;

    /// Computes one iteration of the algorithm.
    fn next_iter(&mut self) -> Result<ArgminIterationData<Self::Parameters>, Error>;
}

/// This trait needs to be implemented by all loggers
pub trait ArgminLog {
    /// Logs general information (a message `msg` and/or key-value pairs `kv`).
    fn log_info(&self, &str, &ArgminKV) -> Result<(), Error>;

    /// Logs information from iterations. Only accepts key-value pairs. `log_iter` is made to log
    /// to a database or a CSV file. Therefore the structure of the key-value pairs should not
    /// change inbetween iterations.
    fn log_iter(&self, &ArgminKV) -> Result<(), Error>;
}

/// Every writer (which is something that writes parameter vectors somewhere after each iteration)
/// needs to implement this.
pub trait ArgminWrite {
    type Param;

    /// Writes the parameter vector somewhere
    fn write(&self, &Self::Param) -> Result<(), Error>;
}

/// TODO: think about removing this.
pub struct ArgminIterationData<T: Clone> {
    param: T,
    cost: f64,
    kv: Option<ArgminKV>,
}

impl<T: Clone> ArgminIterationData<T> {
    pub fn new(param: T, cost: f64) -> Self {
        ArgminIterationData {
            param: param,
            cost: cost,
            kv: None,
        }
    }

    pub fn param(&self) -> T {
        self.param.clone()
    }

    pub fn cost(&self) -> f64 {
        self.cost
    }

    pub fn add_kv(&mut self, kv: ArgminKV) -> &mut Self {
        self.kv = Some(kv);
        self
    }

    pub fn get_kv(&self) -> Option<ArgminKV> {
        self.kv.clone()
    }
}

pub trait ArgminOperator {
    type Parameters;
    type OperatorOutput;

    fn apply(&self, &Self::Parameters) -> Result<Self::OperatorOutput, Error>;

    fn gradient(&self, &Self::Parameters) -> Result<Self::Parameters, Error> {
        Err(ArgminError::NotImplemented {
            text: "Method `gradient` of ArgminOperator trait not implemented!".to_string(),
        }.into())
    }

    // Modifies a parameter vector. Comes with a variable that indicates the "degree" of the
    // modification.
    fn modify(&mut self, &Self::Parameters, f64) -> Result<Self::Parameters, Error> {
        Err(ArgminError::NotImplemented {
            text: "Method `modify` of ArgminOperator trait not implemented!".to_string(),
        }.into())
    }

    fn box_clone(
        &self,
    ) -> Box<ArgminOperator<Parameters = Self::Parameters, OperatorOutput = Self::OperatorOutput>>;
}

impl<T, U> Clone for Box<ArgminOperator<Parameters = T, OperatorOutput = U>> {
    fn clone(&self) -> Box<ArgminOperator<Parameters = T, OperatorOutput = U>> {
        self.box_clone()
    }
}
