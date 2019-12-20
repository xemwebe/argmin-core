// Copyright 2018-2020 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Output parameter vectors to file

use crate::{ArgminKV, ArgminOp, Error, IterState, Observe};
use serde::{Deserialize, Serialize};
use std::default::Default;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

#[derive(Copy, Clone, Debug, Serialize, Deserialize, Eq, PartialEq, Ord, PartialOrd)]
pub enum WriteToFileSerializer {
    Bincode,
    JSON,
}

impl Default for WriteToFileSerializer {
    fn default() -> Self {
        WriteToFileSerializer::Bincode
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq, Ord, PartialOrd)]
pub struct WriteToFile<O> {
    dir: String,
    prefix: String,
    serializer: WriteToFileSerializer,
    _param: std::marker::PhantomData<O>,
}

impl<O: ArgminOp> WriteToFile<O> {
    pub fn new(dir: &str, prefix: &str) -> Self {
        WriteToFile {
            dir: dir.to_string(),
            prefix: prefix.to_string(),
            serializer: WriteToFileSerializer::Bincode,
            _param: std::marker::PhantomData,
        }
    }

    pub fn serializer(mut self, serializer: WriteToFileSerializer) -> Self {
        self.serializer = serializer;
        self
    }
}

impl<O: ArgminOp> Observe<O> for WriteToFile<O> {
    fn observe_iter(&mut self, state: &IterState<O>, _kv: &ArgminKV) -> Result<(), Error> {
        let param = state.get_param();
        let iter = state.get_iter();
        let dir = Path::new(&self.dir);
        if !dir.exists() {
            std::fs::create_dir_all(&dir)?
        }

        let mut fname = self.prefix.clone();
        fname.push_str("_");
        fname.push_str(&iter.to_string());
        fname.push_str(".arp");
        let fname = dir.join(fname);

        let f = BufWriter::new(File::create(fname)?);
        match self.serializer {
            WriteToFileSerializer::Bincode => {
                bincode::serialize_into(f, &param)?;
            }
            WriteToFileSerializer::JSON => {
                serde_json::to_writer_pretty(f, &param)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    send_sync_test!(write_to_file, WriteToFile<Vec<f64>>);
}
