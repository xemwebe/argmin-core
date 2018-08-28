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
// pub use failure::format_err;

/// Error handling
mod errors;
/// Key value datastructure
mod kv;
/// Macros
pub mod macros;
/// Math utilities
mod math;
/// Definition of the return type of the solvers
mod result;
/// Definition of termination reasons
mod termination;

pub use errors::*;
pub use failure::Error;
pub use kv::ArgminKV;
pub use math::*;
pub use result::ArgminResult;
pub use termination::TerminationReason;

/// Every solver must implement this trait
pub trait ArgminSolver {
    /// Type of the parameter vector
    type Parameters;

    /// Computes one iteration of the algorithm.
    fn next_iter(&mut self) -> Result<ArgminIterationData, Error>;

    /// Runs the algorithm. Created by the `make_run!` macro.
    fn run(&mut self) -> Result<ArgminResult<Self::Parameters>, Error>;

    /// Returns the result.
    fn get_result(&self) -> ArgminResult<Self::Parameters>;

    /// Logs the info about the solver. Implemented by the `make_logging!` macro
    fn init_log(&self) -> Result<(), Error>;

    /// Logs key-value data after each iteration. Take care that the key-value stores structure
    /// does not change inbetween iterations, as this can be used to write to rather rigid
    /// formats/storages such as CSV and Databases.
    /// Implemented by the `make_logging!` macro.
    fn log_iter(&self, &ArgminKV) -> Result<(), Error>;

    /// Logs any general information you may have.
    /// Implemented by the `make_logging!` macro.
    fn log_info(&self, &str, &ArgminKV) -> Result<(), Error>;

    /// Export the parameter vector.
    /// Implemented by the `make_write!` macro.
    fn write(&self, &Self::Parameters) -> Result<(), Error>;

    /// Return the current parameter vector
    fn get_param(&self) -> Self::Parameters;

    /// Set the termination reason
    /// Implemented by the `make_terminate!` macro.
    fn set_termination_reason(&mut self, TerminationReason);

    /// Get the termination reaseon
    /// Implemented by the `make_terminate!` macro.
    fn get_termination_reason(&self) -> TerminationReason;

    /// Checks if the algorithm has terminated
    /// Implemented by the `make_terminate!` macro.
    fn terminated(&self) -> bool;

    /// Returns the textual representation of `TerminationReason`
    /// Implemented by the `make_terminate!` macro.
    fn termination_text(&self) -> &str;

    /// Evaluate all stopping criterions and return the `TerminationReason`
    /// Implemented by the `make_terminate!` macro.
    fn terminate(&mut self) -> TerminationReason;
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
pub struct ArgminIterationData {
    kv: Option<ArgminKV>,
}

impl ArgminIterationData {
    pub fn new() -> Self {
        ArgminIterationData { kv: None }
    }

    pub fn add_kv(&mut self, kv: ArgminKV) -> &mut Self {
        self.kv = Some(kv);
        self
    }

    pub fn get_kv(&self) -> Option<ArgminKV> {
        self.kv.clone()
    }
}
