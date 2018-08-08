// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # `ArgminResult`
//!
//! Return type of the solvers. Includes the final parameter vector, the final cost, the number of
//! iterations, whether it terminated and the reason of termination.

use termination::TerminationReason;

/// Return struct for all solvers.
#[derive(Debug, Clone)]
pub struct ArgminResult<T> {
    /// Final parameter vector
    pub param: T,
    /// Final cost value
    pub cost: f64,
    /// Number of iterations
    pub iters: u64,
    /// Indicated whether it terminated or not
    pub terminated: bool,
    /// Reason of termination
    pub termination_reason: TerminationReason,
}

impl<T> ArgminResult<T> {
    /// Constructor
    ///
    /// `param`: Final (best) parameter vector
    /// `cost`: Final (best) cost function value
    /// `iters`: Number of iterations
    /// `termination_reason`: Reason of termination
    pub fn new(param: T, cost: f64, iters: u64, termination_reason: TerminationReason) -> Self {
        ArgminResult {
            param,
            cost,
            iters,
            terminated: termination_reason.terminated(),
            termination_reason: termination_reason,
        }
    }
}
