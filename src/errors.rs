// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Errors

#[derive(Debug, Fail)]
pub enum ArgminError {
    /// Indicates and invalid parameter
    #[fail(display = "Invalid parameter: {}", parameter)]
    InvalidParameter { parameter: String },

    /// Indicates that a function is not implemented
    #[fail(display = "Not implemented: {}", text)]
    NotImplemented { text: String },
}
