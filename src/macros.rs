// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # Macros

/// Creates an `ArgminKV` at compile time in order to avoid pushing to the `kv` vector.
#[macro_export]
macro_rules! make_kv {
    ($($k:expr =>  $v:expr;)*) => {
        ArgminKV { kv: vec![ $(($k, format!("{:?}", $v))),* ] }
    };
}

/// Release an `T` from an `Option<T>` if it is not `None`. If it is `None`, return an
/// `ArgminError` with a message that needs to be provided.
#[macro_export]
macro_rules! check_param {
    ($param:expr, $msg:expr, $error:ident) => {
        match $param {
            None => {
                return Err(ArgminError::$error {
                    text: $msg.to_string(),
                }
                .into());
            }
            Some(ref x) => x.clone(),
        }
    };
    ($param:expr, $msg:expr) => {
        check_param!($param, $msg, NotInitialized);
    };
}
