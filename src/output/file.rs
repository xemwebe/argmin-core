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
    dir: String,
    _param: std::marker::PhantomData<T>,
}

impl<T> WriteToFile<T> {
    pub fn new(dir: &str) -> Arc<Self> {
        Arc::new(WriteToFile {
            dir: dir.to_string(),
            _param: std::marker::PhantomData,
        })
    }
}

impl<T: Serialize + Send + Sync> ArgminWrite for WriteToFile<T> {
    type Param = T;

    fn write(&self, param: &T, iter: u64) -> Result<(), Error> {
        println!("Writing!");

        let dir = Path::new(&self.dir);
        if !dir.exists() {
            std::fs::create_dir_all(&dir)?
        }

        let mut fname = "param_".to_string();
        fname.push_str(&iter.to_string());
        fname.push_str(".arp");
        let fname = dir.join(fname);
        println!("{:?}", fname);

        let f = BufWriter::new(File::create(fname)?);
        bincode::serialize_into(f, param)?;
        Ok(())
    }
}
