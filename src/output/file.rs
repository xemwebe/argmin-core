// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Output parameter vectors to file

use crate::ArgminWrite;
use crate::Error;
use serde::Serialize;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use std::sync::Arc;

pub struct WriteToFile<T> {
    _param: std::marker::PhantomData<T>,
}

impl<T> WriteToFile<T> {
    pub fn new() -> Arc<Self> {
        Arc::new(WriteToFile {
            _param: std::marker::PhantomData,
        })
    }
}

impl<T: Serialize + Send + Sync> ArgminWrite for WriteToFile<T> {
    type Param = T;
    fn write(&self, param: &T) -> Result<(), Error> {
        println!("Writing!");

        // let dir = Path::new(&self.directory);
        // if !dir.exists() {
        //     std::fs::create_dir_all(&dir)?
        // }
        // let fname = dir.join(Path::new(&filename));
        //
        let fname = Path::new("blah.param");

        let f = BufWriter::new(File::create(fname)?);
        bincode::serialize_into(f, param)?;
        Ok(())
    }
}
