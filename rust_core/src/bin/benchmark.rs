// Simple benchmark without external dependencies

use monty_core::{DataLoader, Experiment, ExperimentMode, Model, Observation};
use std::time::Instant;

#[derive(Debug, Clone)]
struct BenchObservation {
    position: [f32; 3],
    feature: f32,
}

impl Observation for BenchObservation {}

struct BenchModel {
    mode: ExperimentMode,
    steps_taken: usize,
    observations_seen: Vec<BenchObservation>,
    confidence: f32,
}

impl BenchModel {
    fn new() -> Self {
        Self {
            mode: ExperimentMode::Train,
            steps_taken: 0,
            observations_seen: Vec::with_capacity(1000),
            confidence: 0.0,
        }
    }
}

impl Model for BenchModel {
    type Obs = BenchObservation;

    fn step(&mut self, observation: &Self::Obs) {
        self.steps_taken += 1;

        match self.mode {
            ExperimentMode::Train => {
                self.observations_seen.push(observation.clone());
            }
            ExperimentMode::Eval => {
                if !self.observations_seen.is_empty() {
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
                }
            }
        }
    }

    fn is_done(&self) -> bool {
        false
    }

    fn pre_episode(&mut self) {
        self.steps_taken = 0;
        self.confidence = 0.0;
    }

    fn post_episode(&mut self) {}

    fn set_experiment_mode(&mut self, mode: ExperimentMode) {
        self.mode = mode;
    }

    fn experiment_mode(&self) -> ExperimentMode {
        self.mode
    }
}

struct BenchDataLoader {
    observations: Vec<BenchObservation>,
    current_index: usize,
}

impl BenchDataLoader {
    fn new(size: usize) -> Self {
        let observations = (0..size)
            .map(|i| {
                let t = i as f32 * 0.1;
                BenchObservation {
                    position: [t.sin(), t.cos(), t * 0.5],
                    feature: t.sin() + t.cos(),
                }
            })
            .collect();

        Self {
            observations,
            current_index: 0,
        }
    }
}

impl DataLoader for BenchDataLoader {
    type Obs = BenchObservation;

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

fn benchmark_experiment(size: usize, epochs: usize) -> (u128, u128) {
    let model = BenchModel::new();
    let train_loader = BenchDataLoader::new(size);
    let eval_loader = BenchDataLoader::new(size);

    let mut experiment = Experiment::new(
        model,
        train_loader,
        eval_loader,
        epochs,
        epochs,
        10000,
        10000,
        true,
        true,
    );

    let start = Instant::now();
    experiment.run();
    let duration = start.elapsed();

    let total_steps = size * epochs * 2; // train + eval
    let steps_per_sec = (total_steps as f64 / duration.as_secs_f64()) as u128;

    (duration.as_micros(), steps_per_sec)
}

fn main() {
    println!("==============================================");
    println!("Monty Core Loop - Performance Benchmarks");
    println!("==============================================\n");

    println!("Running benchmarks (release build)...\n");

    let test_cases = vec![
        (10, 10),
        (50, 10),
        (100, 10),
        (500, 5),
        (1000, 2),
    ];

    println!("{:<15} {:<15} {:<20} {:<20}", "Dataset Size", "Epochs", "Time (ms)", "Steps/sec");
    println!("{:-<70}", "");

    for (size, epochs) in test_cases {
        let (time_us, steps_per_sec) = benchmark_experiment(size, epochs);
        let time_ms = time_us as f64 / 1000.0;

        println!(
            "{:<15} {:<15} {:<20.2} {:<20}",
            size, epochs, time_ms, steps_per_sec
        );
    }

    println!("\n==============================================");
    println!("Memory Characteristics (estimated):");
    println!("==============================================");

    println!("\nPer-observation memory:");
    println!("  BenchObservation: {} bytes", std::mem::size_of::<BenchObservation>());
    println!("  Position [f32; 3]: {} bytes", std::mem::size_of::<[f32; 3]>());
    println!("  Feature f32: {} bytes", std::mem::size_of::<f32>());

    println!("\nModel memory (approximate):");
    println!("  Base struct: ~{} bytes", std::mem::size_of::<BenchModel>());
    println!("  + 1000 observations: ~{} KB",
        (1000 * std::mem::size_of::<BenchObservation>()) / 1024);

    println!("\n==============================================\n");
}
