// Copyright 2025 Thousand Brains Project
//
// Use of this source code is governed by the MIT
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

//! Example demonstrating the minimal Monty core loop in Rust

use monty_core::{DataLoader, Experiment, ExperimentMode, Model, Observation};

/// A simple observation containing a position and feature value
#[derive(Debug, Clone)]
struct SensorObservation {
    position: [f32; 3],
    feature: f32,
}

impl Observation for SensorObservation {}

/// A simple model that learns and matches observations
struct SimpleMontyModel {
    mode: ExperimentMode,
    steps_taken: usize,
    observations_seen: Vec<SensorObservation>,
    confidence: f32,
    confidence_threshold: f32,
}

impl SimpleMontyModel {
    fn new(confidence_threshold: f32) -> Self {
        Self {
            mode: ExperimentMode::Train,
            steps_taken: 0,
            observations_seen: Vec::new(),
            confidence: 0.0,
            confidence_threshold,
        }
    }

    /// Matching step - tries to recognize the object
    fn matching_step(&mut self, observation: &SensorObservation) {
        println!(
            "    [Matching] Processing obs at pos [{:.2}, {:.2}, {:.2}] with feature {:.2}",
            observation.position[0],
            observation.position[1],
            observation.position[2],
            observation.feature
        );

        // Simulate evidence accumulation
        // In real Monty, this would match against learned graph models
        if !self.observations_seen.is_empty() {
            // Simple similarity check
            let similarity = self
                .observations_seen
                .iter()
                .map(|obs| {
                    let dist = ((obs.position[0] - observation.position[0]).powi(2)
                        + (obs.position[1] - observation.position[1]).powi(2)
                        + (obs.position[2] - observation.position[2]).powi(2))
                    .sqrt();
                    let feature_diff = (obs.feature - observation.feature).abs();
                    1.0 / (1.0 + dist + feature_diff)
                })
                .sum::<f32>()
                / self.observations_seen.len() as f32;

            self.confidence = (self.confidence + similarity) / 2.0;
            println!("    Confidence: {:.3}", self.confidence);
        }
    }

    /// Exploratory step - builds the object model
    fn exploratory_step(&mut self, observation: &SensorObservation) {
        println!(
            "    [Exploratory] Learning obs at pos [{:.2}, {:.2}, {:.2}] with feature {:.2}",
            observation.position[0],
            observation.position[1],
            observation.position[2],
            observation.feature
        );

        // Store observation in model
        self.observations_seen.push(observation.clone());
        println!(
            "    Model now contains {} observations",
            self.observations_seen.len()
        );
    }
}

impl Model for SimpleMontyModel {
    type Obs = SensorObservation;

    fn step(&mut self, observation: &Self::Obs) {
        self.steps_taken += 1;

        match self.mode {
            ExperimentMode::Train => {
                // During training, we can do both exploratory and matching
                // Here simplified to just exploratory
                self.exploratory_step(observation);
            }
            ExperimentMode::Eval => {
                // During evaluation, only matching
                self.matching_step(observation);
            }
        }
    }

    fn is_done(&self) -> bool {
        // Done when we've achieved sufficient confidence in eval mode
        if self.mode == ExperimentMode::Eval {
            self.confidence >= self.confidence_threshold
        } else {
            // In training, rely on dataloader exhaustion
            false
        }
    }

    fn pre_episode(&mut self) {
        self.steps_taken = 0;
        self.confidence = 0.0;
        println!("  Model ready for episode in {:?} mode", self.mode);
    }

    fn post_episode(&mut self) {
        println!("  Model completed episode with {} steps", self.steps_taken);
    }

    fn set_experiment_mode(&mut self, mode: ExperimentMode) {
        self.mode = mode;
        println!("Model set to {:?} mode", mode);
    }

    fn experiment_mode(&self) -> ExperimentMode {
        self.mode
    }
}

/// A dataloader that simulates sensory observations of an object
struct ObjectDataLoader {
    object_points: Vec<[f32; 3]>,
    current_index: usize,
}

impl ObjectDataLoader {
    fn new_cube() -> Self {
        // Simulate sampling points on a cube surface
        let mut points = Vec::new();

        // Front face
        for i in 0..3 {
            for j in 0..3 {
                points.push([i as f32 * 0.5, j as f32 * 0.5, 1.0]);
            }
        }

        // Back face
        for i in 0..3 {
            for j in 0..3 {
                points.push([i as f32 * 0.5, j as f32 * 0.5, 0.0]);
            }
        }

        Self {
            object_points: points,
            current_index: 0,
        }
    }

    fn new_sphere() -> Self {
        // Simulate sampling points on a sphere surface
        let mut points = Vec::new();

        for theta in 0..6 {
            for phi in 0..6 {
                let theta_rad = theta as f32 * std::f32::consts::PI / 6.0;
                let phi_rad = phi as f32 * std::f32::consts::PI / 3.0;

                let x = theta_rad.sin() * phi_rad.cos();
                let y = theta_rad.sin() * phi_rad.sin();
                let z = theta_rad.cos();

                points.push([x, y, z]);
            }
        }

        Self {
            object_points: points,
            current_index: 0,
        }
    }
}

impl DataLoader for ObjectDataLoader {
    type Obs = SensorObservation;

    fn next_observation(&mut self) -> Option<Self::Obs> {
        if self.current_index < self.object_points.len() {
            let pos = self.object_points[self.current_index];
            let obs = SensorObservation {
                position: pos,
                // Feature based on position (simplified)
                feature: (pos[0] + pos[1] + pos[2]) / 3.0,
            };
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

fn main() {
    println!("==============================================");
    println!("Monty Core Loop - Rust Implementation");
    println!("==============================================\n");

    println!("This demonstrates the minimal sensorimotor learning loop");
    println!("ported from the Python Monty implementation.\n");

    // Create the model
    let model = SimpleMontyModel::new(0.6);

    // Create dataloaders for training and evaluation
    let train_dataloader = ObjectDataLoader::new_cube();
    let eval_dataloader = ObjectDataLoader::new_cube();

    // Create the experiment
    let mut experiment = Experiment::new(
        model,
        train_dataloader,
        eval_dataloader,
        2,   // n_train_epochs
        1,   // n_eval_epochs
        50,  // max_train_steps
        30,  // max_eval_steps
        true, // do_train
        true, // do_eval
    );

    // Run the full experiment
    experiment.run();

    println!("\n==============================================");
    println!("Experiment Complete!");
    println!("==============================================");
}
