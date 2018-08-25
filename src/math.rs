// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Math
//!
//! TODO: Documentation.

pub trait ArgminDot<T, U> {
    fn dot(&self, T) -> U;
}

impl ArgminDot<Vec<f64>, f64> for Vec<f64> {
    fn dot(&self, other: Vec<f64>) -> f64 {
        self.iter().zip(other.iter()).map(|(a, b)| a * b).sum()
    }
}

impl<'a> ArgminDot<&'a Vec<f64>, f64> for Vec<f64> {
    fn dot(&self, other: &'a Vec<f64>) -> f64 {
        self.iter().zip(other.iter()).map(|(a, b)| a * b).sum()
    }
}

pub trait ArgminSub<T> {
    fn sub(&self, T) -> Self;
}

impl ArgminSub<Vec<f64>> for Vec<f64> {
    fn sub(&self, other: Vec<f64>) -> Vec<f64> {
        self.iter().zip(other.iter()).map(|(a, b)| a - b).collect()
    }
}

impl<'a> ArgminSub<&'a Vec<f64>> for Vec<f64> {
    fn sub(&self, other: &'a Vec<f64>) -> Vec<f64> {
        self.iter().zip(other.iter()).map(|(a, b)| a - b).collect()
    }
}

pub trait ArgminScaledAdd<T, U> {
    fn scaled_add(&self, U, T) -> Self;
}

impl ArgminScaledAdd<Vec<f64>, f64> for Vec<f64> {
    fn scaled_add(&self, scale: f64, other: Vec<f64>) -> Vec<f64> {
        self.iter()
            .zip(other.iter())
            .map(|(a, b)| a + scale * b)
            .collect()
    }
}

impl<'a> ArgminScaledAdd<&'a Vec<f64>, f64> for Vec<f64> {
    fn scaled_add(&self, scale: f64, other: &'a Vec<f64>) -> Vec<f64> {
        self.iter()
            .zip(other.iter())
            .map(|(a, b)| a + scale * b)
            .collect()
    }
}

pub trait ArgminScaledSub<T, U> {
    fn scaled_sub(&self, U, T) -> Self;
}

impl ArgminScaledSub<Vec<f64>, f64> for Vec<f64> {
    fn scaled_sub(&self, scale: f64, other: Vec<f64>) -> Vec<f64> {
        self.iter()
            .zip(other.iter())
            .map(|(a, b)| a - scale * b)
            .collect()
    }
}

impl<'a> ArgminScaledSub<&'a Vec<f64>, f64> for Vec<f64> {
    fn scaled_sub(&self, scale: f64, other: &Vec<f64>) -> Vec<f64> {
        self.iter()
            .zip(other.iter())
            .map(|(a, b)| a + scale * b)
            .collect()
    }
}
