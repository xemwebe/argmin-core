// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Argmin Optimizaton toolbox

pub mod problem;
/// Definition of the return type of the solvers
pub mod result;

use result::ArgminResult;

pub trait ArgminNextIter {
    fn next_iter(&mut self);
}

pub trait ArgminGetResult {
    type ParamVec;
    fn get_result(&self) -> ArgminResult<Self::ParamVec>;
}

pub trait ArgminRun {
    type ParamVec;
    fn run(&mut self) -> ArgminResult<Self::ParamVec>;
}

// extern crate ndarray;
// extern crate ndarray_linalg;
// extern crate num;
// extern crate rand;
//
// /// Traits for implementing parameter vectors
// pub mod parameter;
//
//
// use num::{Bounded, ToPrimitive};
// use parameter::ArgminParameter;
// use std::default::Default;
// use std::fmt::Debug;
//
// /// Trait for cost function values
// /// TODO: Do this with trait aliases once they work in rust.
// pub trait ArgminCostValue:
//     Bounded + ToPrimitive + Copy + Debug + Default + PartialOrd + Send + Sync
// {
// }
// impl<T> ArgminCostValue for T where
//     T: Bounded + ToPrimitive + Copy + Debug + Default + PartialOrd + Send + Sync
// {}
//
// /// Trait every solve needs to implement (in the future)
// pub trait ArgminSolver<'a> {
//     /// Parameter vector
//     type Parameter: ArgminParameter;
//     /// Cost value
//     type CostValue: ArgminCostValue;
//     /// Hessian
//     type Hessian;
//     /// Initial parameter(s)
//     type StartingPoints: Send + Sync;
//     /// Type of Problem
//     type ProblemDefinition: Clone + Send;
//
//     /// Initializes the solver and sets the state to its initial state
//     // fn init(&mut self, &'a Self::ProblemDefinition, &Self::StartingPoints) -> Result<()>;
//     fn init(&mut self, Self::ProblemDefinition, &Self::StartingPoints) -> Result<()>;
//
//     /// Moves forward by a single iteration
//     fn next_iter(&mut self) -> Result<ArgminResult<Self::Parameter, Self::CostValue>>;
//
//     /// Run initialization and iterations at once
//     fn run(
//         &mut self,
//         // &'a Self::ProblemDefinition,
//         Self::ProblemDefinition,
//         &Self::StartingPoints,
//     ) -> Result<ArgminResult<Self::Parameter, Self::CostValue>>;
//
//     /// Handles the stopping criteria
//     fn terminate(&self) -> TerminationReason;
// }
