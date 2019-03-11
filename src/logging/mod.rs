// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Logging
//!
//! Provides logging functionality for solvers.

pub mod slog_logger;

use crate::{ArgminKV, Error, Observe};
use std::sync::Arc;

/// Container for Observers
#[derive(Clone, Default)]
pub struct Observer {
    /// Vector of boxed types which implement `ArgminLog`
    logger: Vec<Arc<Observe>>,
}

impl Observer {
    /// Constructor
    pub fn new() -> Self {
        Observer { logger: vec![] }
    }

    /// Push another `ArgminLog` to the `logger` field
    pub fn push(&mut self, logger: Arc<Observe>) -> &mut Self {
        self.logger.push(logger);
        self
    }
}

/// By implementing `ArgminLog` for `ArgminLogger` we basically allow a set of `ArgminLog`gers to
/// be used just like a single `ArgminLog`ger.
impl Observe for Observer {
    /// Log general info
    fn log_info(&self, msg: &str, kv: &ArgminKV) -> Result<(), Error> {
        for l in self.logger.iter() {
            l.log_info(msg, kv)?
        }
        Ok(())
    }

    /// This should be used to log iteration data only (because this is what may be saved in a CSV
    /// file or a database)
    fn log_iter(&self, kv: &ArgminKV) -> Result<(), Error> {
        for l in self.logger.iter() {
            l.log_iter(kv)?
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    send_sync_test!(argmin_logger, ArgminLogger);
}
