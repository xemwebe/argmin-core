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
use crate::ArgminOp;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

/// Return struct for all solvers.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ArgminResult<O: ArgminOp> {
    /// Final parameter vector
    pub param: O::Param,
    /// Final cost value
    pub cost: f64,
    /// Number of iterations
    pub iters: u64,
    /// Indicated whether it terminated or not
    pub terminated: bool,
    /// Reason of termination
    pub termination_reason: TerminationReason,
    /// operator
    pub operator: O,
    /// total time
    pub total_time: std::time::Duration,
}

impl<O: ArgminOp> ArgminResult<O> {
    /// Constructor
    ///
    /// `param`: Final (best) parameter vector
    /// `cost`: Final (best) cost function value
    /// `iters`: Number of iterations
    /// `termination_reason`: Reason of termination
    pub fn new(
        param: O::Param,
        cost: f64,
        iters: u64,
        termination_reason: TerminationReason,
        operator: O,
        total_time: std::time::Duration,
    ) -> Self {
        ArgminResult {
            param,
            cost,
            iters,
            terminated: termination_reason.terminated(),
            termination_reason,
            operator,
            total_time,
        }
    }
}

impl<O> std::fmt::Display for ArgminResult<O>
where
    O: ArgminOp,
    O::Param: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        writeln!(f, "ArgminResult:")?;
        writeln!(f, "    param:       {:?}", self.param)?;
        writeln!(f, "    cost:        {}", self.cost)?;
        writeln!(f, "    iters:       {}", self.iters)?;
        writeln!(f, "    termination: {}", self.termination_reason)?;
        writeln!(f, "    time:        {:?}", self.total_time)?;
        Ok(())
    }
}

impl<O: ArgminOp> PartialEq for ArgminResult<O> {
    fn eq(&self, other: &ArgminResult<O>) -> bool {
        (self.cost - other.cost).abs() < std::f64::EPSILON
    }
}

impl<O: ArgminOp> Eq for ArgminResult<O> {}

impl<O: ArgminOp> Ord for ArgminResult<O> {
    fn cmp(&self, other: &ArgminResult<O>) -> Ordering {
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

impl<O: ArgminOp> PartialOrd for ArgminResult<O> {
    fn partial_cmp(&self, other: &ArgminResult<O>) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MinimalNoOperator;

    send_sync_test!(argmin_result, ArgminResult<MinimalNoOperator>);
}
