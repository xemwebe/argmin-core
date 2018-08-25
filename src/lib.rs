// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Argmin Optimizaton toolbox
//!
//! TODO: Documentation.

pub extern crate failure;
#[macro_use]
pub extern crate failure_derive;
// pub use failure::format_err;

// /// Errors
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

pub trait ArgminSolver {
    type Parameters;

    fn next_iter(&mut self) -> Result<ArgminIterationData, Error>;
    fn run(&mut self) -> Result<ArgminResult<Self::Parameters>, Error>;
    fn get_result(&self) -> ArgminResult<Self::Parameters>;

    fn init_log(&self);
    fn log_iter(&self, &ArgminKV);
    fn log_info(&self, &str, &ArgminKV);

    fn get_param(&self) -> Self::Parameters;

    fn set_termination_reason(&mut self, TerminationReason);
    fn get_termination_reason(&self) -> TerminationReason;
    fn terminated(&self) -> bool;
    fn termination_text(&self) -> &str;
    fn terminate(&mut self) -> TerminationReason;
}

#[derive(Clone)]
pub enum ArgminParameterOutputTrigger {
    OutputCurrent,
    OutputBest,
}

pub trait ArgminLog {
    fn log_info(&self, &str, &ArgminKV);
    fn log_iter(&self, &ArgminKV);
}

// pub trait ArgminWrite {
//     fn write<T>(&self, T);
// }

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
