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

    pub fn merge(&self, other: &ArgminKV) -> ArgminKV {
        // there must be a way which requires less cloning
        let mut kv = self.kv.clone();
        kv.append(&mut other.kv.clone());
        ArgminKV { kv }
    }
}
