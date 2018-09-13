// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Key Value storage
//!
//! A very simple key-value storage.
//!
//! TODOs:
//!   * Either use something existing, or at least evaluate the performance and if necessary,
//!     improve performance.

use std;

/// A simple key-value storage
#[derive(Clone)]
pub struct ArgminKV {
    /// The actual key value storage
    pub kv: Vec<(&'static str, String)>,
}

impl ArgminKV {
    /// Constructor
    pub fn new() -> Self {
        ArgminKV { kv: vec![] }
    }

    /// Push a key-value pair to the `kv` vector.
    ///
    /// This formats the `val` using `format!`. Therefore `T` has to implement `Display`.
    pub fn push<T: std::fmt::Display>(&mut self, key: &'static str, val: T) -> &mut Self {
        self.kv.push((key, format!("{}", val)));
        self
    }

    /// Merge another `kv` into `self.kv`
    ///
    /// TODO: Probably not used anymore (?)
    pub fn merge(&mut self, other: &mut ArgminKV) -> &mut Self {
        self.kv.append(&mut other.kv);
        self
    }
}
