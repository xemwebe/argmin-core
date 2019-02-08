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
//!
//! TODO:
//!   * Maybe it is more appropriate to return the `base` struct?

use crate::termination::TerminationReason;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

/// Return struct for all solvers.
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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
            termination_reason,
        }
    }
}

impl<T: std::fmt::Debug> std::fmt::Display for ArgminResult<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "ArgminResult:\n")?;
        write!(f, "    param:       {:?}\n", self.param)?;
        write!(f, "    cost:        {}\n", self.cost)?;
        write!(f, "    iters:       {}\n", self.iters)?;
        write!(f, "    termination: {}\n", self.termination_reason)?;
        Ok(())
    }
}

impl<T> PartialEq for ArgminResult<T> {
    fn eq(&self, other: &ArgminResult<T>) -> bool {
        (self.cost - other.cost).abs() < std::f64::EPSILON
    }
}

impl<T> Eq for ArgminResult<T> {}

impl<T> Ord for ArgminResult<T> {
    fn cmp(&self, other: &ArgminResult<T>) -> Ordering {
        let t = self.cost - other.cost;
        if t.abs() < std::f64::EPSILON {
            Ordering::Equal
        } else if t > 0.0 {
            Ordering::Greater
        } else {
            Ordering::Less
        }
    }
}

impl<T> PartialOrd for ArgminResult<T> {
    fn partial_cmp(&self, other: &ArgminResult<T>) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    send_sync_test!(argmin_result, ArgminResult<Vec<f64>>);
}
