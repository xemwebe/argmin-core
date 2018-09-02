// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Key Value storage

use std;

#[derive(Clone)]
pub struct ArgminKV {
    pub kv: Vec<(&'static str, String)>,
}

impl ArgminKV {
    pub fn new() -> Self {
        ArgminKV { kv: vec![] }
    }

    pub fn push<T: std::fmt::Display>(&mut self, key: &'static str, val: T) -> &mut Self {
        self.kv.push((key, format!("{}", val)));
        self
    }

    pub fn merge(&mut self, other: &mut ArgminKV) -> &mut Self {
        self.kv.append(&mut other.kv);
        self
    }
}
