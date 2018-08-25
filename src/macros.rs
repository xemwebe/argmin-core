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
            let total_time = std::time::Instant::now();

            self.init_log()?;

            let running = Arc::new(AtomicBool::new(true));
            let r = running.clone();

            ctrlc::set_handler(move || {
                r.store(false, Ordering::SeqCst);
            })?;

            while running.load(Ordering::SeqCst) {
                let start = std::time::Instant::now();

                let mut data = self.next_iter()?;

                let duration = start.elapsed();

                // only log if there is something to log
                if let Some(ref mut log) = data.get_kv() {
                    log.push("time", duration.as_secs() as f64 + duration.subsec_nanos() as f64 * 1e-9);
                    self.log_iter(&log)?;
                }

                self.terminate();
                if self.terminated() {
                    break;
                }
            }

            // in case it stopped prematurely and `termination_reason` is still `NotTerminated`,
            // someone must have pulled the handbrake
            if self.iter < self.max_iters && self.termination_reason == TerminationReason::NotTerminated
            {
                self.set_termination_reason(TerminationReason::Aborted);
            }

            let duration = total_time.elapsed();

            let mut kv = ArgminKV::new();
            let kv = make_kv!(
                "termination_reason" => self.get_termination_reason();
                "total_time" => duration.as_secs() as f64 + duration.subsec_nanos() as f64 * 1e-9;
            );
            // kv.push("total_time", duration.as_secs() as f64 + duration.subsec_nanos() as f64 * 1e-9);

            self.log_info(
                &format!("Terminated: {reason}", reason = self.termination_text(),),
                &kv,
            )?;

            Ok(self.get_result())
        }
    }
}

/// This macro generates the `terminate` function for every solver which implements `ArgminSolver`.
#[macro_export]
macro_rules! make_terminate {
    ($self:ident; $condition:expr, $reason:path;) => {
        if $condition {
            $self.set_termination_reason($reason);
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
            $self.set_termination_reason(TerminationReason::NotTerminated);
            TerminationReason::NotTerminated
        }

        fn set_termination_reason(&mut $self, termination_reason: TerminationReason) {
            $self.termination_reason = termination_reason;
        }

        fn get_termination_reason(&$self) -> TerminationReason {
            $self.termination_reason.clone()
        }

        fn terminated(&$self) -> bool {
            $self.termination_reason.terminated()
        }

        fn termination_text(&$self) -> &str {
            $self.termination_reason.text()
        }
    };
}

#[macro_export]
macro_rules! make_logging {
    ($self:ident, $msg:expr, $($k:expr =>  $v:expr;)*) => {
        fn init_log(&$self) -> Result<(), Error> {
            let logs = make_kv!($($k => $v;)*);
            $self.logger.log_info($msg, &logs)
        }

        fn log_iter(&$self, kv: &ArgminKV) -> Result<(), Error> {
            $self.logger.log_iter(kv)
        }

        fn log_info(&$self, msg: &str, kv: &ArgminKV) -> Result<(), Error> {
            $self.logger.log_info(msg, kv)
        }
    };
}

#[macro_export]
macro_rules! make_kv {
    ($($k:expr =>  $v:expr;)*) => {
        ArgminKV { kv: vec![ $(($k, format!("{:?}", $v))),* ] }
    };
}
