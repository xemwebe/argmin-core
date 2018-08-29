// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Macros

/// Creates the `run` function which should work when `ArgminSolver` is properly implemented.
#[macro_export]
macro_rules! make_run {
    () => {
        fn run(&mut self) -> Result<ArgminResult<Self::Parameters>, Error> {
            use ctrlc;
            use std::sync::atomic::{AtomicBool, Ordering};
            use std::sync::Arc;

            let total_time = std::time::Instant::now();

            // do the inital logging
            self.init_log()?;

            // Set up the Ctrl-C handler
            let running = Arc::new(AtomicBool::new(true));
            let r = running.clone();
            ctrlc::set_handler(move || {
                r.store(false, Ordering::SeqCst);
            })?;

            while running.load(Ordering::SeqCst) {
                // execute iteration
                let start = std::time::Instant::now();
                let mut data = self.next_iter()?;
                let duration = start.elapsed();

                // increment iteration number
                self.base.increment_iter();

                // Set new current parameter
                self.base.set_cur_param(data.param())
                         .set_cur_cost(data.cost());

                // check if parameters are the best so far
                if data.cost() < self.base.best_cost() {
                    self.base.set_best_param(data.param())
                             .set_best_cost(data.cost());
                }

                // only log if there is something to log
                if let Some(ref mut log) = data.get_kv() {
                    log.push("time", duration.as_secs() as f64 + duration.subsec_nanos() as f64 * 1e-9);
                    self.base.log_iter(&log)?;
                }

                // Write to file or something
                self.base.write(&self.base.cur_param());

                self.terminate();
                if self.base.terminated() {
                    break;
                }
            }

            // in case it stopped prematurely and `termination_reason` is still `NotTerminated`,
            // someone must have pulled the handbrake
            if self.base.cur_iter() < self.base.max_iters() && !self.base.terminated() {
                self.base.set_termination_reason(TerminationReason::Aborted);
            }

            self.base.set_total_time(total_time.elapsed());

            let mut kv = ArgminKV::new();
            let kv = make_kv!(
                "termination_reason" => self.base.termination_reason();
                "total_time" => self.base.total_time().as_secs() as f64 +
                                self.base.total_time().subsec_nanos() as f64 * 1e-9;
            );

            self.base.log_info(
                &format!("Terminated: {reason}", reason = self.base.termination_reason_text(),),
                &kv,
            )?;

            Ok(self.base.result())
        }

        fn result(&self) -> ArgminResult<Self::Parameters> {
            self.base.result()
        }

        fn set_max_iters(&mut self, iters: u64) {
            self.base.set_max_iters(iters);
        }

        fn add_logger(&mut self, logger: Box<ArgminLog>) {
            self.base.add_logger(logger);
        }

        fn add_writer(&mut self, writer: Box<ArgminWrite<Param = Self::Parameters>>) {
            self.base.add_writer(writer);
        }
    }
}

/// This macro generates the `terminate` function for every solver which implements `ArgminSolver`.
#[macro_export]
macro_rules! make_terminate {
    ($self:ident; $condition:expr, $reason:path;) => {
        if $condition {
            $self.base.set_termination_reason($reason);
            return $reason;
        }
    };
    ($self:ident; $condition:expr, $reason:path; $($x:expr, $y:path;)*) => {
            make_terminate!($self; $condition, $reason;);
            make_terminate!($self; $($x, $y;)*);
    };
    ($self:ident, $($x:expr => $y:path;)*) => {
        fn terminate(&mut $self) -> TerminationReason {
            make_terminate!($self; $($x, $y;)*);
            $self.base.set_termination_reason(TerminationReason::NotTerminated);
            TerminationReason::NotTerminated
        }
    };
}

#[macro_export]
macro_rules! make_logging {
    ($self:ident, $msg:expr, $($k:expr =>  $v:expr;)*) => {
        fn init_log(&$self) -> Result<(), Error> {
            let logs = make_kv!($($k => $v;)*);
            $self.base.log_info($msg, &logs)
        }
    };
}

#[macro_export]
macro_rules! make_kv {
    ($($k:expr =>  $v:expr;)*) => {
        ArgminKV { kv: vec![ $(($k, format!("{:?}", $v))),* ] }
    };
}
