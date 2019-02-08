// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::ArgminSolver;
use serde::{Deserialize, Serialize};

pub trait ArgminSnapshot {
    fn store(&self);
    fn load() -> Self;
}

impl<'de, T> ArgminSnapshot for T
where
    T: ArgminSolver + Serialize + Deserialize<'de>,
{
    fn store(&self) {
        unimplemented!();
    }

    fn load() -> Self {
        unimplemented!();
    }
}
