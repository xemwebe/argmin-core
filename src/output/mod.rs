// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Write parameter vectors somewhere
//!
//! This is very basic and needs much more work. The idea is that parameter vectors are written to
//! disk (or anywhere, essentially) on a regular basis, or on certain events (new best parameter
//! vector found for instance).
//! Several standard implementations should be provided (like writing an image as a PNG) as well as
//! the possibility to implement such a writer for custom types. This will require a bit of
//! thinking.

pub mod file;

use serde::{Deserialize, Serialize};
use std::default::Default;

#[derive(Copy, Clone, Serialize, Deserialize, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum WriterMode {
    Never,
    Always,
    Every(u64),
    NewBest,
}

impl Default for WriterMode {
    fn default() -> WriterMode {
        WriterMode::Always
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//
//     send_sync_test!(argmin_write, ArgminWriter<Vec<f64>>);
// }
