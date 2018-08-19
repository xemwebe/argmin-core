// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Argmin Optimizaton toolbox
//!
//! TODO: Documentation.

/// Key value datastructure
mod kv;
/// Macros
pub mod macros;
/// Definition of the return type of the solvers
mod result;
/// Definition of termination reasons
mod termination;

pub use kv::ArgminKV;
pub use result::ArgminResult;
pub use termination::TerminationReason;

pub trait ArgminNextIter {
    fn next_iter(&mut self) -> ArgminKV;
}

pub trait ArgminGetResult {
    type Parameters;
    fn get_result(&self) -> ArgminResult<Self::Parameters>;
}

pub trait ArgminRun {
    type Parameters;
    fn run(&mut self) -> ArgminResult<Self::Parameters>;
}

pub trait ArgminLog {
    fn log_info(&self, &str, &ArgminKV);
    fn log_iter(&self, &ArgminKV);
}

pub trait ArgminInitLog {
    fn init_log(&self);
}
