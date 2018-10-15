// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Errors
//!
//! Predefined errors.
//!
//! TODOs:
//!   * Provide an `impl` to make it easier to create such errors.

#[derive(Debug, Fail)]
pub enum ArgminError {
    /// Indicates and invalid parameter
    ///
    /// TODO: Rename `parameter` to `text` and fix this in all the other code as well.
    #[fail(display = "Invalid parameter: {}", parameter)]
    InvalidParameter { parameter: String },

    /// Indicates that a function is not implemented
    #[fail(display = "Not implemented: {}", text)]
    NotImplemented { text: String },

    /// Indicates that a function is not initialized
    #[fail(display = "Not initialized: {}", text)]
    NotInitialized { text: String },

    /// Indicates that a condition is violated
    #[fail(display = "Condition violated: {}", text)]
    ConditionViolated { text: String },

    /// Indicates an impossible error
    #[fail(display = "Impossible Error: {}", text)]
    ImpossibleError { text: String },
}
