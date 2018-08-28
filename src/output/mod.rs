// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Logging

pub mod file;

use ArgminWrite;
use Error;

pub struct ArgminWriter<T> {
    writers: Vec<Box<ArgminWrite<Param = T>>>,
}

impl<T> ArgminWriter<T> {
    pub fn new() -> Self {
        ArgminWriter { writers: vec![] }
    }

    pub fn push(&mut self, writer: Box<ArgminWrite<Param = T>>) -> &mut Self {
        self.writers.push(writer);
        self
    }
}

impl<T: Clone> ArgminWrite for ArgminWriter<T> {
    type Param = T;
    fn write(&self, param: &T) -> Result<(), Error> {
        for w in self.writers.iter() {
            w.write(&param)?;
        }
        Ok(())
    }
}
