// Copyright 2025 Thousand Brains Project
//
// Use of this source code is governed by the MIT
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

//! Monty Core - Minimal sensorimotor learning loop
//!
//! This is a Rust implementation of the core loop from the Monty
//! sensorimotor learning system (Python version).
//!
//! The core loop structure:
//! - run() -> train() -> run_epoch() -> run_episode() -> model.step()
//!
//! This minimal implementation focuses on the essential structure
//! without the full complexity of the Python version.

use std::fmt::Debug;

/// Represents a single observation from the environment/dataloader
pub trait Observation: Debug + Clone {}

/// Trait for the Model that processes observations
///
/// This corresponds to the Monty class in abstract_monty_classes.py
pub trait Model {
    type Obs: Observation;

    /// Take a step with the given observation
    /// This can be either a matching step or exploratory step
    fn step(&mut self, observation: &Self::Obs);

    /// Check if the model has finished processing (terminal condition)
    fn is_done(&self) -> bool;

    /// Hook called before an episode starts
    fn pre_episode(&mut self);

    /// Hook called after an episode ends
    fn post_episode(&mut self);

    /// Set the experiment mode (train or eval)
    fn set_experiment_mode(&mut self, mode: ExperimentMode);

    /// Get the current experiment mode
    fn experiment_mode(&self) -> ExperimentMode;
}

/// Trait for data loaders that provide observations
pub trait DataLoader {
    type Obs: Observation;

    /// Get the next observation
    /// Returns None when the episode is complete
    fn next_observation(&mut self) -> Option<Self::Obs>;

    /// Hook called before an episode starts
    fn pre_episode(&mut self);

    /// Hook called before an epoch starts
    fn pre_epoch(&mut self);

    /// Reset the dataloader for a new episode
    fn reset(&mut self);
}

/// Experiment mode (train or eval)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExperimentMode {
    Train,
    Eval,
}

/// The main experiment that orchestrates the training/evaluation loop
///
/// This corresponds to MontyExperiment in monty_experiment.py
pub struct Experiment<M, D>
where
    M: Model,
    D: DataLoader<Obs = M::Obs>,
{
    model: M,
    train_dataloader: D,
    eval_dataloader: D,
    n_train_epochs: usize,
    n_eval_epochs: usize,
    max_train_steps: usize,
    max_eval_steps: usize,
    do_train: bool,
    do_eval: bool,
}

impl<M, D> Experiment<M, D>
where
    M: Model,
    D: DataLoader<Obs = M::Obs>,
{
    /// Create a new experiment
    pub fn new(
        model: M,
        train_dataloader: D,
        eval_dataloader: D,
        n_train_epochs: usize,
        n_eval_epochs: usize,
        max_train_steps: usize,
        max_eval_steps: usize,
        do_train: bool,
        do_eval: bool,
    ) -> Self {
        Self {
            model,
            train_dataloader,
            eval_dataloader,
            n_train_epochs,
            n_eval_epochs,
            max_train_steps,
            max_eval_steps,
            do_train,
            do_eval,
        }
    }

    /// Run the full experiment (train and/or eval)
    ///
    /// Corresponds to run() in frameworks/run.py
    pub fn run(&mut self) {
        if self.do_train {
            println!("---------training---------");
            self.train();
        }

        if self.do_eval {
            println!("---------evaluating---------");
            self.evaluate();
        }
    }

    /// Run training epochs
    ///
    /// Corresponds to train() in monty_experiment.py:556
    pub fn train(&mut self) {
        self.pre_train();
        self.model.set_experiment_mode(ExperimentMode::Train);

        let n_epochs = self.n_train_epochs;
        let max_steps = self.max_train_steps;

        for epoch in 0..n_epochs {
            println!("Training epoch {}/{}", epoch + 1, n_epochs);
            self.train_dataloader.pre_epoch();
            self.run_episode_train(max_steps);
            self.post_epoch();
        }

        self.post_train();
    }

    /// Run evaluation epochs
    ///
    /// Corresponds to evaluate() in monty_experiment.py:564
    pub fn evaluate(&mut self) {
        self.pre_eval();
        self.model.set_experiment_mode(ExperimentMode::Eval);

        let n_epochs = self.n_eval_epochs;
        let max_steps = self.max_eval_steps;

        for epoch in 0..n_epochs {
            println!("Evaluation epoch {}/{}", epoch + 1, n_epochs);
            self.eval_dataloader.pre_epoch();
            self.run_episode_eval(max_steps);
            self.post_epoch();
        }

        self.post_eval();
    }

    /// Run one training episode until model.is_done or max_steps
    ///
    /// This is the CORE LOOP - corresponds to run_episode() in monty_experiment.py:458
    fn run_episode_train(&mut self, max_steps: usize) {
        self.train_dataloader.pre_episode();
        self.model.pre_episode();

        let mut step = 0;
        while let Some(observation) = self.train_dataloader.next_observation() {
            self.pre_step(step, &observation);

            // CORE STEP: Process the observation
            self.model.step(&observation);

            self.post_step(step, &observation);

            step += 1;

            // Check terminal conditions
            if self.model.is_done() || step >= max_steps {
                break;
            }
        }

        println!("Episode complete after {} steps", step);
        self.model.post_episode();
    }

    /// Run one evaluation episode until model.is_done or max_steps
    fn run_episode_eval(&mut self, max_steps: usize) {
        self.eval_dataloader.pre_episode();
        self.model.pre_episode();

        let mut step = 0;
        while let Some(observation) = self.eval_dataloader.next_observation() {
            self.pre_step(step, &observation);

            // CORE STEP: Process the observation
            self.model.step(&observation);

            self.post_step(step, &observation);

            step += 1;

            // Check terminal conditions
            if self.model.is_done() || step >= max_steps {
                break;
            }
        }

        println!("Episode complete after {} steps", step);
        self.model.post_episode();
    }

    // Lifecycle hooks

    fn pre_train(&self) {
        println!("Starting training...");
    }

    fn post_train(&self) {
        println!("Training complete.");
    }

    fn pre_eval(&self) {
        println!("Starting evaluation...");
    }

    fn post_eval(&self) {
        println!("Evaluation complete.");
    }

    fn post_epoch(&self) {
        // Hook for post-epoch processing
    }

    fn pre_step(&self, step: usize, _observation: &M::Obs) {
        if step % 10 == 0 {
            println!("  Step {}", step);
        }
    }

    fn post_step(&self, _step: usize, _observation: &M::Obs) {
        // Hook for post-step processing
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Simple test observation
    #[derive(Debug, Clone)]
    struct TestObservation {
        value: f32,
    }

    impl Observation for TestObservation {}

    // Simple test model
    struct TestModel {
        steps_taken: usize,
        max_steps_to_done: usize,
        mode: ExperimentMode,
    }

    impl TestModel {
        fn new(max_steps_to_done: usize) -> Self {
            Self {
                steps_taken: 0,
                max_steps_to_done,
                mode: ExperimentMode::Train,
            }
        }
    }

    impl Model for TestModel {
        type Obs = TestObservation;

        fn step(&mut self, observation: &Self::Obs) {
            self.steps_taken += 1;
            println!("    Model processing observation: {:?}", observation);
        }

        fn is_done(&self) -> bool {
            self.steps_taken >= self.max_steps_to_done
        }

        fn pre_episode(&mut self) {
            self.steps_taken = 0;
        }

        fn post_episode(&mut self) {
            // Reset for next episode
        }

        fn set_experiment_mode(&mut self, mode: ExperimentMode) {
            self.mode = mode;
        }

        fn experiment_mode(&self) -> ExperimentMode {
            self.mode
        }
    }

    // Simple test dataloader
    struct TestDataLoader {
        observations: Vec<TestObservation>,
        current_index: usize,
    }

    impl TestDataLoader {
        fn new(num_observations: usize) -> Self {
            let observations = (0..num_observations)
                .map(|i| TestObservation {
                    value: i as f32 * 0.1,
                })
                .collect();

            Self {
                observations,
                current_index: 0,
            }
        }
    }

    impl DataLoader for TestDataLoader {
        type Obs = TestObservation;

        fn next_observation(&mut self) -> Option<Self::Obs> {
            if self.current_index < self.observations.len() {
                let obs = self.observations[self.current_index].clone();
                self.current_index += 1;
                Some(obs)
            } else {
                None
            }
        }

        fn pre_episode(&mut self) {
            self.current_index = 0;
        }

        fn pre_epoch(&mut self) {
            self.current_index = 0;
        }

        fn reset(&mut self) {
            self.current_index = 0;
        }
    }

    #[test]
    fn test_core_loop() {
        let model = TestModel::new(5);
        let train_loader = TestDataLoader::new(10);
        let eval_loader = TestDataLoader::new(10);

        let mut experiment = Experiment::new(
            model,
            train_loader,
            eval_loader,
            1, // n_train_epochs
            1, // n_eval_epochs
            100, // max_train_steps
            100, // max_eval_steps
            true, // do_train
            true, // do_eval
        );

        experiment.run();
    }
}
