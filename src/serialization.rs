// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::Error;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::default::Default;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

#[derive(Clone, Serialize, Deserialize, Eq, PartialEq, Ord, PartialOrd, Debug, Hash, Copy)]
pub enum CheckpointMode {
    Never,
    Every(u64),
    Always,
}

impl Default for CheckpointMode {
    fn default() -> CheckpointMode {
        CheckpointMode::Never
    }
}

#[derive(Clone, Serialize, Deserialize, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
pub struct ArgminCheckpoint {
    mode: CheckpointMode,
    directory: String,
    prefix: String,
}

impl Default for ArgminCheckpoint {
    fn default() -> ArgminCheckpoint {
        let out = ArgminCheckpoint::new("checkpoints", CheckpointMode::default());
        match out {
            Ok(cp) => cp,
            Err(_) => panic!("Cannot create default ArgminCheckpoint."),
        }
    }
}

impl ArgminCheckpoint {
    pub fn new(directory: &str, mode: CheckpointMode) -> Result<Self, Error> {
        match mode {
            CheckpointMode::Every(_) | CheckpointMode::Always => {
                std::fs::create_dir_all(&directory)?
            }
            _ => {}
        }
        let prefix = "solver".to_string();
        let directory = directory.to_string();
        Ok(ArgminCheckpoint {
            mode,
            directory,
            prefix,
        })
    }

    #[inline]
    pub fn dir(&self) -> String {
        self.directory.clone()
    }

    #[inline]
    pub fn set_prefix(&mut self, prefix: &str) {
        self.prefix = prefix.to_string();
    }

    #[inline]
    pub fn prefix(&self) -> String {
        self.prefix.clone()
    }

    #[inline]
    pub fn store<T: Serialize>(&self, solver: &T) -> Result<(), Error> {
        let mut filename = self.prefix();
        filename.push_str(".arg");
        let dir = Path::new(&self.directory).join(Path::new(&filename));
        let f = BufWriter::new(File::create(dir)?);
        // serde_json::to_writer_pretty(f, solver)?;
        bincode::serialize_into(f, solver)?;
        // serde_json::to_string_pretty(self).unwrap()
        Ok(())
    }

    #[inline]
    pub fn store_cond<T: Serialize>(&self, solver: &T, iter: u64) -> Result<(), Error> {
        match self.mode {
            CheckpointMode::Always => self.store(solver)?,
            CheckpointMode::Every(it) if iter % it == 0 => self.store(solver)?,
            CheckpointMode::Never | CheckpointMode::Every(_) => {}
        };
        Ok(())
    }
}

pub fn load_checkpoint<T: DeserializeOwned, P: AsRef<Path>>(path: P) -> Result<T, Error> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    // Ok(serde_json::from_reader(reader)?)
    Ok(bincode::deserialize_from(reader)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nooperator::MinimalNoOperator;
    use crate::*;
    use argmin_codegen::ArgminSolver;
    use std::fmt::Debug;

    #[derive(ArgminSolver, Serialize, Deserialize, Clone, Debug)]
    pub struct PhonySolver<T, H, O>
    where
        T: Clone + Default + Debug,
        H: Clone + Default + Debug,
        O: ArgminOperator<Parameters = T, OperatorOutput = f64, Hessian = H>,
    {
        base: ArgminBase<T, H, O>,
    }

    impl<T, H, O> PhonySolver<T, H, O>
    where
        T: Clone + Default + Debug,
        H: Clone + Default + Debug,
        O: ArgminOperator<Parameters = T, OperatorOutput = f64, Hessian = H>,
    {
        /// Constructor
        pub fn new(op: O, init_param: T) -> Self {
            PhonySolver {
                base: ArgminBase::new(op, init_param),
            }
        }
    }

    impl<T, H, O> ArgminNextIter for PhonySolver<T, H, O>
    where
        T: Clone + Default + Debug,
        H: Clone + Default + Debug,
        O: ArgminOperator<Parameters = T, OperatorOutput = f64, Hessian = H>,
    {
        type Parameters = T;
        type OperatorOutput = f64;
        type Hessian = H;

        fn next_iter(&mut self) -> Result<ArgminIterationData<Self::Parameters>, Error> {
            unimplemented!()
        }
    }

    #[test]
    fn test_store() {
        let op: MinimalNoOperator = MinimalNoOperator::new();
        let solver = PhonySolver::new(op, vec![0.0, 0.0]);
        let check = ArgminCheckpoint::new("checkpoints", CheckpointMode::Always).unwrap();
        check.store_cond(&solver, 20).unwrap();

        let loaded: PhonySolver<Vec<f64>, Vec<Vec<f64>>, MinimalNoOperator> =
            load_checkpoint("checkpoints/solver.arg").unwrap();
        println!("{:?}", loaded);
    }
}