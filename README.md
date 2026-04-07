# neural-net

Small feed-forward neural network package with three layers of functionality:

- `neural_net.backend`: the raw FFNN implementation and math helpers
- `neural_net.training`: a higher-level training API for datasets and scalar functions
- `neural_net.demo`: the interactive `sin(x)` demo with milestone snapshots on a slider

The project is packaged for `uv`, so it can be run locally as a script or imported elsewhere as a module.

## Setup

Install the environment and dependencies:

```bash
uv sync
```

Run the demo either through the compatibility wrapper or the package entrypoint:

```bash
uv run learn_function_demo.py
uv run neural-net-demo
```

## Demo

The demo trains a small network to approximate `sin(x)` on `[-pi, pi]`.

It:

- builds a random `1 -> 32 -> 32 -> 1` network
- trains for milestone counts `2^n`
- evaluates the network on a dense grid
- opens a Matplotlib plot with a slider at the bottom
- saves the current figure to `sin_learning_milestones.png`

Useful commands:

```bash
uv run learn_function_demo.py
uv run learn_function_demo.py --verbose
uv run learn_function_demo.py --runs 12
uv run neural-net-demo --verbose --runs 15
```

CLI options:

- `-v`, `--verbose`: print milestone progress during training without blocking the hot loop
- `-n`, `--runs`: choose the largest exponent `n`; training records snapshots at `2^0, 2^1, ..., 2^n`

Example:

- `--runs 6` records snapshots after `1, 2, 4, 8, 16, 32, 64` updates
- `--runs 12` records snapshots up to `4096` updates

## Demo Configuration

The current demo configuration lives in `neural_net/demo.py`.

By default it uses:

- activation: `ActivationFunc.tanh`
- network shape: `(32, 32, 1)`
- training config: `TrainingConfig(max_power=args.runs)`
- target function: `np.sin`

If you want different behavior, the simplest options are:

- change the network shape passed to `build_random_network(...)`
- change the activation function
- change the target function from `np.sin` to another scalar function
- change `TrainingConfig` values such as `learning_rate`, `max_power`, `evaluation_points`, or `seed`

## Package API

Top-level imports are re-exported from `neural_net/__init__.py`, so the common API can be imported directly:

```python
from neural_net import (
    ActivationFunc,
    FFNN,
    FFNNConfig,
    TrainingConfig,
    TrainingResult,
    build_random_network,
    fit_dataset,
    fit_function,
    predict_dataset,
)
```

If you want lower-level control, import directly from `neural_net.backend` or `neural_net.training`.

## Use The Training Backend With Your Own Data

Use `fit_dataset(...)` when you already have input/output samples.

Expected shapes:

- inputs: `(n_samples, input_dim)`
- targets: `(n_samples, output_dim)`

For scalar inputs or outputs, 1D arrays are also accepted and normalized internally.

Example:

```python
import numpy as np

from neural_net import TrainingConfig, build_random_network, fit_dataset, predict_dataset
from neural_net.backend import ActivationFunc

x_train = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
], dtype=float)

y_train = np.array([
    [0.0],
    [1.0],
    [1.0],
    [0.0],
], dtype=float)

network = build_random_network(
    input_layer_dim=2,
    hidden_layer_shapes=(16, 16, 1),
    activation=ActivationFunc.tanh,
    seed=0,
)

result = fit_dataset(
    network,
    x_train,
    y_train,
    config=TrainingConfig(
        learning_rate=0.02,
        max_power=10,
        evaluation_points=128,
        seed=0,
    ),
)

print(result.milestone_steps)
print(result.losses)

predictions = predict_dataset(network, x_train)
print(predictions)
```

What `fit_dataset(...)` does:

- samples one training row at a time
- runs `fast_backward_pass(...)` on that sample
- records predictions and loss at milestone steps `2^n`
- returns a `TrainingResult`

`TrainingResult` contains:

- `evaluation_inputs`: the inputs used for milestone evaluation
- `evaluation_targets`: the reference outputs for those inputs
- `snapshots`: a dictionary `{step: predictions}`
- `losses`: a dictionary `{step: mse}`
- `network`: the trained network instance
- `milestone_steps`: the tuple of recorded milestone counts

## Train A Function f: R -> R

Use `fit_function(...)` when your target is a scalar function and you want the trainer to sample from a domain.

Example:

```python
import numpy as np

from neural_net import TrainingConfig, build_random_network, fit_function
from neural_net.backend import ActivationFunc

network = build_random_network(
    input_layer_dim=1,
    hidden_layer_shapes=(32, 32, 1),
    activation=ActivationFunc.tanh,
    seed=0,
)

result = fit_function(
    network,
    np.sin,
    domain=(-np.pi, np.pi),
    config=TrainingConfig(
        learning_rate=0.02,
        max_power=12,
        evaluation_points=512,
        seed=0,
    ),
)

print(result.losses)
```

Current `fit_function(...)` assumptions:

- input dimension must be `1`
- output dimension must be `1`
- the target function must accept NumPy inputs and return numeric outputs

## Use The Backend Directly

If you want to bypass the training helpers and work directly with the network:

```python
import numpy as np

from neural_net.backend import ActivationFunc, FFNN, FFNNConfig

config = FFNNConfig(
    input_layer_dim=1,
    hidden_layer_count=3,
    hidden_layer_shapes=(8, 8, 1),
    activation_func=ActivationFunc.tanh,
)

network = FFNN(config)
output = network.fast_forward_pass(np.array([0.5]))
print(output)
```

The training helpers are usually the better entrypoint unless you are experimenting with the network internals.

## Reuse In Another Project

Because this is a normal Python package, you can install it into another environment and import from `neural_net`.

Within this repo, the main reusable modules are:

- `neural_net/backend.py`
- `neural_net/training.py`
- `neural_net/demo.py`

The top-level files `learn_function_demo.py`, `main.py`, and `training_framework.py` are compatibility wrappers rather than the primary implementation.
