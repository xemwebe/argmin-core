// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Termination
//!
//! Defines reasons for termination.
//!
//! TODO:
//!   * Maybe it is better to define a trait (with `terminated` and `text` methods), because it
//!     would allow implementers of solvers to define their own `TerminationReason`s. However, this
//!     would require a lot of work.

use serde::{Deserialize, Serialize};

/// Indicates why the optimization algorithm stopped
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TerminationReason {
    /// In case it has not terminated yet
    NotTerminated,
    /// Maximum number of iterations reached
    MaxItersReached,
    /// Target cost function value reached
    TargetCostReached,
    /// Target precision reached
    TargetPrecisionReached,
    /// Acceped stall iter exceeded
    AcceptedStallIterExceeded,
    /// Best stall iter exceeded
    BestStallIterExceeded,
    /// Condition for Line search met
    LineSearchConditionMet,
    /// Aborted
    Aborted,
}

impl TerminationReason {
    /// Returns `true` if a solver terminated and `false` otherwise
    pub fn terminated(&self) -> bool {
        match *self {
            TerminationReason::NotTerminated => false,
            _ => true,
        }
    }

    /// Returns a texual representation of what happened
    ///
    /// TODO: I am not sure if this is the best way to solve this.
    pub fn text(&self) -> &str {
        match *self {
            TerminationReason::NotTerminated => "Not terminated",
            TerminationReason::MaxItersReached => "Maximum number of iterations reached",
            TerminationReason::TargetCostReached => "Target cost value reached",
            TerminationReason::TargetPrecisionReached => "Target precision reached",
            TerminationReason::AcceptedStallIterExceeded => "Accepted stall iterations exceeded",
            TerminationReason::BestStallIterExceeded => "Best stall iterations exceeded",
            TerminationReason::LineSearchConditionMet => "Line search condition met",
            TerminationReason::Aborted => "Optimization aborted",
        }
    }
}
