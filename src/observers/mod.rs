// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Logging
//!
//! Provides logging functionality for solvers.

pub mod file;
pub mod slog_logger;

use crate::{ArgminKV, ArgminOp, Error, IterState, Observe};
use serde::{Deserialize, Serialize};
use std::default::Default;
use std::sync::Arc;

pub use file::*;
pub use slog_logger::*;

/// Container for Observers
#[derive(Clone, Default)]
pub struct Observer<O> {
    /// Vector of boxed types which implement `ArgminLog`
    logger: Vec<(Arc<Observe<O>>, ObserverMode)>,
}

impl<O: ArgminOp> Observer<O> {
    /// Constructor
    pub fn new() -> Self {
        Observer { logger: vec![] }
    }

    /// Push another `ArgminLog` to the `logger` field
    pub fn push<OBS: Observe<O> + 'static>(
        &mut self,
        observer: OBS,
        mode: ObserverMode,
    ) -> &mut Self {
        self.logger.push((Arc::new(observer), mode));
        self
    }
}

/// By implementing `ArgminLog` for `ArgminLogger` we basically allow a set of `ArgminLog`gers to
/// be used just like a single `ArgminLog`ger.
impl<O: ArgminOp> Observe<O> for Observer<O> {
    /// Log general info
    fn observe_init(&self, msg: &str, kv: &ArgminKV) -> Result<(), Error> {
        for l in self.logger.iter() {
            l.0.observe_init(msg, kv)?
        }
        Ok(())
    }

    /// This should be used to log iteration data only (because this is what may be saved in a CSV
    /// file or a database)
    fn observe_iter(&self, state: &IterState<O>, kv: &ArgminKV) -> Result<(), Error> {
        use ObserverMode::*;
        for l in self.logger.iter() {
            let iter = state.get_iter();
            match l.1 {
                Always => l.0.observe_iter(state, kv),
                Every(i) if iter % i == 0 => l.0.observe_iter(state, kv),
                NewBest if state.is_best() => l.0.observe_iter(state, kv),
                Never | Every(_) | NewBest => Ok(()),
            }?
        }
        Ok(())
    }
}

#[derive(Copy, Clone, Serialize, Deserialize, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum ObserverMode {
    Never,
    Always,
    Every(u64),
    NewBest,
}

impl Default for ObserverMode {
    fn default() -> ObserverMode {
        ObserverMode::Always
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    send_sync_test!(argmin_logger, ArgminLogger);
}
