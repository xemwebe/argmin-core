// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Output parameter vectors to file

use std;
use ArgminWrite;
use Error;

pub struct WriteToFile<T> {
    _param: std::marker::PhantomData<T>,
}

impl<T> WriteToFile<T> {
    pub fn new() -> Box<Self> {
        Box::new(WriteToFile {
            _param: std::marker::PhantomData,
        })
    }
}

impl<T> ArgminWrite for WriteToFile<T> {
    type Param = T;
    fn write(&self, _param: &T) -> Result<(), Error> {
        println!("Writing!");
        Ok(())
    }
}
