// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Output parameter vectors to file

use crate::ArgminWrite;
use crate::Error;
use crate::WriterMode;
use serde::{Deserialize, Serialize};
use std::default::Default;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

#[derive(Copy, Clone, Serialize, Deserialize, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum WriteToFileSerializer {
    Bincode,
    JSON,
}

impl Default for WriteToFileSerializer {
    fn default() -> Self {
        WriteToFileSerializer::Bincode
    }
}

#[derive(Clone, Serialize, Deserialize, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct WriteToFile<T> {
    dir: String,
    prefix: String,
    mode: WriterMode,
    serializer: WriteToFileSerializer,
    _param: std::marker::PhantomData<T>,
}

impl<T: Serialize + Send + Sync> WriteToFile<T> {
    pub fn new(dir: &str, prefix: &str) -> Self {
        WriteToFile {
            dir: dir.to_string(),
            prefix: prefix.to_string(),
            mode: WriterMode::default(),
            serializer: WriteToFileSerializer::Bincode,
            _param: std::marker::PhantomData,
        }
    }

    pub fn set_serializer(&mut self, serializer: WriteToFileSerializer) -> &mut Self {
        self.serializer = serializer;
        self
    }

    pub fn set_mode(&mut self, mode: WriterMode) -> &mut Self {
        self.mode = mode;
        self
    }

    fn write_to_file(&self, param: &T, iter: u64) -> Result<(), Error> {
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
                bincode::serialize_into(f, param)?;
            }
            WriteToFileSerializer::JSON => {
                serde_json::to_writer_pretty(f, param)?;
            }
        }
        Ok(())
    }
}

impl<T: Serialize + Send + Sync> ArgminWrite for WriteToFile<T> {
    type Param = T;

    fn write(&self, param: &T, iter: u64, new_best: bool) -> Result<(), Error> {
        use WriterMode::*;
        match self.mode {
            Always => self.write_to_file(param, iter),
            Every(i) if iter % i == 0 => self.write_to_file(param, iter),
            NewBest if new_best => self.write_to_file(param, iter),
            Never | Every(_) | NewBest => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    send_sync_test!(write_to_file, WriteToFile<Vec<f64>>);
}
