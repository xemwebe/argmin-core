// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::nooperator::NoOperator;
use crate::ArgminSolver;
use serde::{Deserialize, Serialize};
use std::default::Default;
use std::fmt::Debug;

pub trait ArgminSnapshot {
    fn store(&self) -> String;
    fn load() -> Self;
}

impl<'de, T> ArgminSnapshot for T
where
    T: Serialize + Deserialize<'de>,
{
    fn store(&self) -> String {
        serde_json::to_string(self).unwrap()
    }

    fn load() -> Self {
        unimplemented!();
    }
}

// impl<'de, T, U, H> ArgminSnapshot for NoOperator<T, U, H>
// where
//     T: Clone + Default + Debug + Send + Sync + Serialize + Deserialize<'de>,
//     U: Clone + Default + Debug + Send + Sync + Serialize + Deserialize<'de>,
//     H: Clone + Default + Debug + Send + Sync + Serialize + Deserialize<'de>,
// {
//     fn store(&self) -> String {
//         serde_json::to_string(self).unwrap()
//     }
//
//     fn load() -> Self {
//         unimplemented!();
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nooperator::NoOperator;

    #[test]
    fn test_store() {
        let op: NoOperator<Vec<f64>, f64, Vec<Vec<f64>>> = NoOperator::new();
        println!("{:#?}", op.store());
    }
}
