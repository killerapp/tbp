# Monty Core Loop - Rust Implementation

This directory contains a Rust implementation of the minimal core loop from the Monty sensorimotor learning system (Python version).

## Overview

The Monty system is a neuroscience-inspired sensorimotor learning framework from the Thousand Brains Project. This Rust implementation focuses on the **minimal core loop** that drives the training and evaluation process.

## Core Loop Structure

The implementation mirrors the Python architecture:

```
run() -> train()/evaluate() -> run_epoch() -> run_episode() -> model.step()
```

### Python Mapping

| Python File | Rust Module | Description |
|-------------|-------------|-------------|
| `frameworks/run.py:run()` | `Experiment::run()` | Top-level experiment runner |
| `frameworks/experiments/monty_experiment.py:train()` | `Experiment::train()` | Training loop |
| `frameworks/experiments/monty_experiment.py:evaluate()` | `Experiment::evaluate()` | Evaluation loop |
| `frameworks/experiments/monty_experiment.py:run_episode()` | `Experiment::run_episode_*()` | Core episode loop |
| `frameworks/models/abstract_monty_classes.py:_matching_step()` | `Model::step()` (Eval mode) | Matching step |
| `frameworks/models/abstract_monty_classes.py:_exploratory_step()` | `Model::step()` (Train mode) | Exploratory step |

## Architecture

### Core Traits

1. **`Observation`**: Represents a single sensory observation
   - Trait marker for observation data types
   - Must implement `Debug` and `Clone`

2. **`Model`**: The learning/inference model
   - `step()`: Process one observation (core algorithm)
   - `is_done()`: Check terminal condition
   - `pre_episode()` / `post_episode()`: Episode lifecycle hooks
   - `set_experiment_mode()` / `experiment_mode()`: Mode management

3. **`DataLoader`**: Provides observations
   - `next_observation()`: Get next observation (iterator-like)
   - `pre_epoch()` / `pre_episode()`: Setup hooks
   - `reset()`: Reset for new episode

4. **`Experiment`**: Orchestrates the training/evaluation loop
   - `run()`: Main entry point
   - `train()`: Run training epochs
   - `evaluate()`: Run evaluation epochs
   - Internal episode management with lifecycle hooks

### Key Differences from Python

1. **Type Safety**: Rust's type system enforces correct usage at compile time
2. **Ownership**: Explicit ownership and borrowing rules prevent data races
3. **Performance**: Zero-cost abstractions and no GC overhead
4. **Simplified**: Focus on core loop structure without full Python complexity

## Building and Running

### Build
```bash
cd rust_core
cargo build
```

### Run Tests
```bash
cargo test
```

### Run Demo
```bash
cargo run
```

The demo shows:
- **Training phase**: Model learns object structure (exploratory steps)
- **Evaluation phase**: Model recognizes the object (matching steps)

## Example Output

```
==============================================
Monty Core Loop - Rust Implementation
==============================================

---------training---------
Starting training...
Model set to Train mode
Training epoch 1/2
  Model ready for episode in Train mode
  Step 0
    [Exploratory] Learning obs at pos [0.00, 0.00, 1.00] with feature 0.33
    Model now contains 1 observations
    ...

---------evaluating---------
Starting evaluation...
Model set to Eval mode
Evaluation epoch 1/1
  Model ready for episode in Eval mode
  Step 0
    [Matching] Processing obs at pos [0.00, 0.00, 1.00] with feature 0.33
    Confidence: 0.230
    ...
```

## Code Structure

```
rust_core/
├── Cargo.toml              # Package configuration
├── README.md               # This file
└── src/
    ├── lib.rs              # Core loop implementation (traits + Experiment)
    └── main.rs             # Demo with SimpleMontyModel and ObjectDataLoader
```

## Implementation Details

### Core Loop (src/lib.rs)

The central loop in `Experiment::run_episode_*()`:

```rust
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

    self.model.post_episode();
}
```

This directly corresponds to Python's `monty_experiment.py:run_episode()`:

```python
def run_episode(self):
    self.pre_episode()
    for step, observation in enumerate(self.dataloader):
        self.pre_step(step, observation)
        self.model.step(observation)  # CORE STEP
        self.post_step(step, observation)
        if self.model.is_done or step >= self.max_steps:
            break
    self.post_episode(step)
```

### Demo Implementation (src/main.rs)

The demo includes:
- **`SimpleMontyModel`**: A simplified model that:
  - Stores observations during training (exploratory steps)
  - Matches against stored observations during eval (matching steps)
  - Accumulates confidence for recognition

- **`ObjectDataLoader`**: Simulates sampling points on object surfaces
  - `new_cube()`: Sample points on a cube
  - `new_sphere()`: Sample points on a sphere (unused in demo)

## Future Extensions

To fully implement the Python Monty system, you would add:

1. **Graph-based memory**: Spatial graphs for object models
2. **Evidence-based matching**: Bayesian hypothesis management
3. **Motor system**: Action selection and goal state generation
4. **Voting system**: Lateral communication between learning modules
5. **Hierarchical processing**: Multi-level learning module connections
6. **Habitat integration**: 3D environment simulation
7. **Logging and visualization**: Experiment tracking and plotting

## License

Copyright 2025 Thousand Brains Project

Use of this source code is governed by the MIT license that can be found in the LICENSE file or at https://opensource.org/licenses/MIT.

## References

- Python Implementation: `/home/user/tbp/src/tbp/monty/frameworks/`
- Core Loop: `monty_experiment.py:run_episode()` (line 458)
- Model Interface: `abstract_monty_classes.py` (lines 21-43)
- TBP Website: https://thousandbrainsproject.org/
- GitHub: https://github.com/thousandbrainsproject/tbp.monty
