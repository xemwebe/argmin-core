// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Macros

/// This macro generates the `terminate` function for every solver which implements `ArgminSolver`.
#[macro_export]
macro_rules! make_terminate {
    ($self:ident => $condition:expr, $reason:path;) => {
        if $condition {
            $self.set_termination_reason($reason);
            return $reason;
        }
    };
    ($self:ident => $condition:expr, $reason:path ; $($x: expr, $y:path;)*) => {
            make_terminate!($self => $condition, $reason;);
            make_terminate!($self => $($x, $y;)*);
    };
    ($self:ident, $($x: expr, $y:path;)*) => {
        fn terminate(&mut $self) -> TerminationReason {
            make_terminate!($self => $($x, $y;)*);
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
