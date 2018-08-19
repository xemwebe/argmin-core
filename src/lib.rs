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
pub mod kv;
/// Macros
pub mod macros;
/// Definition of the return type of the solvers
pub mod result;
/// Definition of termination reasons
pub mod termination;

pub use kv::ArgminKV;

use result::ArgminResult;

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
