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
        fn run(&mut self) -> ArgminResult<Self::Parameters> {
            self.init_log();

            let running = Arc::new(AtomicBool::new(true));
            let r = running.clone();

            ctrlc::set_handler(move || {
                r.store(false, Ordering::SeqCst);
            }).expect("Error setting Ctrl-C handler!");

            while running.load(Ordering::SeqCst) {
                let log = self.next_iter();

                self.log_iter(&log);

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

            self.log_info(
                &format!("Terminated: {reason}", reason = self.termination_text(),),
                &ArgminKV::new(),
            );

            self.get_result()
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
    ($l:ident; $kk:expr => $vv:expr;) => {
        $l.push($kk, $vv);
    };
    ($l:ident; $kk:expr => $vv:expr; $($k:expr =>  $v:expr;)*) => {
            make_logging!($l; $kk => $vv;);
            make_logging!($l; $($k => $v;)*);
    };
    ($self:ident, $($k:expr =>  $v:expr;)*) => {
        fn init_log(&$self) {
            let mut logs = ArgminKV::new();
            make_logging!(logs; $($k => $v;)*);
            $self.logger.log_info("blah", &logs);
        }

        fn log_iter(&$self, kv: &ArgminKV) {
            $self.logger.log_iter(kv)
        }

        fn log_info(&$self, msg: &str, kv: &ArgminKV) {
            $self.logger.log_info(msg, kv)
        }
    };
}
