// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! # `ArgminProblem`
//!
//! Struct which holds all information necessary to describe an optimization problem.
//!
// //! # Example
// //!
// //! ```rust
// //! extern crate argmin;
// //! extern crate ndarray;
// //! use ndarray::{Array1, Array2};
// //! use argmin::prelude::*;
// //! use argmin::ArgminProblem;
// //! use argmin::testfunctions::{rosenbrock_derivative_nd, rosenbrock_hessian_nd, rosenbrock_nd};
// //!
// //! // Define cost function
// //! let cost = |x: &Array1<f64>| -> f64 { rosenbrock_nd(x, 1_f64, 100_f64) };
// //! let gradient = |x: &Array1<f64>| -> Array1<f64> { rosenbrock_derivative_nd(x, 1_f64, 100_f64) };
// //! let hessian = |x: &Array1<f64>| -> Array2<f64> { rosenbrock_hessian_nd(x, 1_f64, 100_f64) };
// //!
// //! // Set up problem
// //! // The problem requires a cost function, gradient and hessian.
// //! let mut prob = ArgminProblem::new(&cost);
// //! prob.gradient(&gradient);
// //! prob.hessian(&hessian);
// //! ```

// use errors::*;
// use prelude::*;
// use parameter::ArgminParameter;

/// This struct hold all information that describes the optimization problem.
//#[derive(Clone)]
pub struct ArgminProblem<'a> {
    /// reference to a function which computes the cost/fitness for a given parameter vector
    pub cost_function: &'a Fn(Vec<f64>) -> f64,
    // /// optional reference to a function which provides the gradient at a given point in parameter
    // /// space
    // pub gradient: Option<Box<Fn(Vec<f64>) -> Vec<f64>>>,
    // /// optional reference to a function which provides the Hessian at a given point in parameter
    // /// space
    // pub hessian: Option<Box<Fn(Vec<f64>) -> Vec<Vec<f64>>>>,
    /// lower bound of the parameter vector
    pub lower_bound: Option<Vec<f64>>,
    /// upper bound of the parameter vector
    pub upper_bound: Option<Vec<f64>>,
    // /// (non)linear constraint which is `true` if a parameter vector lies within the bounds
    // pub constraint: &'a Fn(&T) -> bool,
    /// Target cost function value. The optimization will stop once this value is reached.
    pub target_cost: f64,
}

impl<'a> ArgminProblem<'a> {
    /// Create a new `ArgminProblem` struct.
    ///
    /// The field `gradient` is automatically set to `None`, but can be manually set by the
    /// `gradient` function. The (non) linear constraint `constraint` is set to a closure which
    /// evaluates to `true` everywhere. This can be overwritten with the `constraint` function.
    ///
    /// `cost_function`: Reference to a cost function
    pub fn new(cost_function: &'a Fn(Vec<f64>) -> f64) -> Self {
        ArgminProblem {
            cost_function: cost_function,
            // gradient: None,
            // hessian: None,
            lower_bound: None,
            upper_bound: None,
            // constraint: &|_x: &T| true,
            // target_cost: std::f64::min_value(),
            target_cost: 0.0,
        }
    }

    /// Set lower and upper bounds
    ///
    /// `lower_bound`: lower bound for the parameter vector
    /// `upper_bound`: upper bound for the parameter vector
    pub fn bounds(&mut self, lower_bound: Vec<f64>, upper_bound: Vec<f64>) -> &mut Self {
        self.lower_bound = Some(lower_bound);
        self.upper_bound = Some(upper_bound);
        self
    }

    // /// Provide the gradient
    // ///
    // /// The function has to have the signature `&Fn(&T) -> T` where `T` is the type of
    // /// the parameter vector. The function returns the gradient for a given parameter vector.
    // pub fn gradient(&mut self, gradient: &'a Fn(&T) -> T) -> &mut Self {
    // self.gradient = Some(gradient);
    // self
    // }

    // /// Provide the Hessian
    // ///
    // /// The function has to have the signature `&Fn(&T) -> T` where `T` is the type of
    // /// the parameter vector. The function returns the gradient for a given parameter vector.
    // pub fn hessian(&mut self, hessian: &'a Fn(&T) -> V) -> &mut Self {
    // self.hessian = Some(hessian);
    // self
    // }

    // /// Provide additional (non) linear constraint.
    // ///
    // /// The function has to have the signature `&Fn(&T) -> bool` where `T` is the type of
    // /// the parameter vector. The function returns `true` if all constraints are satisfied and
    // /// `false` otherwise.
    // pub fn constraint(&mut self, constraint: &'a Fn(&T) -> bool) -> &mut Self {
    // self.constraint = constraint;
    // self
    // }

    /// Set target cost function value
    ///
    /// If the optimization reaches this value, it will be stopped.
    pub fn target_cost(&mut self, target_cost: f64) -> &mut Self {
        self.target_cost = target_cost;
        self
    }

    // /// Create a random parameter vector
    // ///
    // /// The parameter vector satisfies the `lower_bound` and `upper_bound`.
    // pub fn random_param(&self) -> Result<Vec<f64>> {
    // match (self.lower_bound.as_ref(), self.upper_bound.as_ref()) {
    // (Some(l), Some(u)) => Ok(Vec<f64>::random(l, u)),
    // _ => panic!("Parameter: lower_bound and upper_bound must be provided."),
    // }
    // }
}
