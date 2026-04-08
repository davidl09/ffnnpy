# neural-net

Small feed-forward neural network package with two training backends:

- `neural_net.backend` and `neural_net.training`: the original scalar reference implementation
- `neural_net.accelerated`: a batched high-throughput backend with NumPy kernels and an optional Numba runtime
- `neural_net.demo`: the interactive `sin(x)` demo with milestone snapshots on a slider

The reference path stays intact for experimentation and teaching. The accelerated path is the one to use when throughput matters.

## Setup

Install the base environment with the NumPy accelerated runtime:

```bash
uv sync
```

To enable the optional compiled Numba runtime for the accelerated backend, install the accelerated extra:

```bash
uv sync --extra accelerated
```

Run the demo either through the compatibility wrapper or the package entrypoint:

```bash
uv run learn_function_demo.py
uv run neural-net-demo
```

## Demo

The demo trains a small network to approximate `sin(x)` on `[-pi, pi]`.

Reference backend:

```bash
uv run neural-net-demo --backend reference --runs 15
```

Accelerated backend with batched NumPy:

```bash
uv run neural-net-demo --backend accelerated --runtime numpy --batch-size 256 --runs 15
```

Accelerated backend with automatic runtime selection:

```bash
uv run neural-net-demo --backend accelerated --runtime auto --batch-size 256 --runs 15
```

CLI options:

- `-v`, `--verbose`: print milestone progress during training
- `-n`, `--runs`: choose the largest exponent `n`; training records snapshots at `2^0, 2^1, ..., 2^n`
- `--backend`: choose `reference` or `accelerated`
- `--runtime`: choose `auto`, `numpy`, or `numba` for the accelerated backend
- `--batch-size`: mini-batch size for accelerated training

The saved plot still goes to `sin_learning_milestones.png`, and the accelerated demo annotates the chart with runtime and batch metadata.

## Package API

Top-level imports are re-exported from `neural_net/__init__.py`:

```python
from neural_net import (
    AcceleratedFFNN,
    AcceleratedRuntime,
    AcceleratedTrainingConfig,
    ActivationFunc,
    FFNN,
    FFNNConfig,
    LoadedNetworkArtifact,
    LossFunc,
    TrainingConfig,
    TrainingResult,
    build_accelerated_network,
    build_random_network,
    fit_dataset,
    fit_dataset_accelerated,
    fit_function,
    fit_function_accelerated,
    load_network,
    predict_dataset,
    predict_dataset_accelerated,
    register_output_modifier,
    save_network,
)
```

The high-level builders accept `loss_func=...`. Supported values are `LossFunc.mse` and
`LossFunc.cross_entropy`. Cross-entropy requires a sigmoid output layer and targets in `[0, 1]`.

## Model Persistence

Save a trained network to a compressed `.ffnnpy` file and load it back later:

```python
from neural_net import (
    TrainingConfig,
    build_random_network,
    load_network,
    save_network,
)

network = build_random_network(
    input_layer_dim=1,
    hidden_layer_shapes=(32, 32, 1),
    activation="tanh",
    seed=0,
)

save_network(
    network,
    "sin_approximator.ffnnpy",
    training_config=TrainingConfig(max_power=12, seed=0),
)

artifact = load_network("sin_approximator.ffnnpy")
restored_network = artifact.network
restored_training_config = artifact.training_config
```

`.ffnnpy` archives store:

- backend kind (`FFNN` or `AcceleratedFFNN`)
- model architecture, activations, loss, and accelerated runtime when relevant
- exact weights and biases
- optional `TrainingConfig` or `AcceleratedTrainingConfig`

If you use an inference-only `output_modifier`, register it under a stable name before saving or loading:

```python
import numpy as np

from neural_net import register_output_modifier


def boolean_threshold(output: np.ndarray) -> bool:
    return bool(output[0] >= 0.5)


register_output_modifier("boolean_threshold", boolean_threshold)
```

## Reference Backend

Use the original backend when you want the simple scalar training path:

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
    config=TrainingConfig(max_power=12, seed=0),
)
```

This path performs one-sample updates and keeps the older `FFNN.fast_forward_pass(...)` and `FFNN.fast_backward_pass(...)` workflow unchanged. `fit_function(...)` accepts either scalar Python callables such as `math.sin` or NumPy-vectorized callables such as `np.sin`.

You can optionally attach an inference-only output modifier when you build the network:

```python
network = build_random_network(
    input_layer_dim=1,
    hidden_layer_shapes=(32, 32, 1),
    activation=ActivationFunc.sigmoid,
    seed=0,
    output_modifier=lambda output: bool(output[0] >= 0.5),
)
```

The modifier receives one sample's final-layer output vector and changes only inference-facing APIs such as `fast_forward_pass(...)` and `predict_dataset(...)`. Training, loss evaluation, and `TrainingResult.snapshots` continue to use the raw numeric outputs.

## Accelerated Backend

Use the accelerated backend when you want vectorized training and inference:

```python
import numpy as np

from neural_net import (
    AcceleratedRuntime,
    AcceleratedTrainingConfig,
    ActivationFunc,
    build_accelerated_network,
    fit_function_accelerated,
)

network = build_accelerated_network(
    input_layer_dim=1,
    hidden_layer_shapes=(32, 32, 1),
    activation=ActivationFunc.tanh,
    seed=0,
    runtime=AcceleratedRuntime.auto,
)

result = fit_function_accelerated(
    network,
    np.sin,
    config=AcceleratedTrainingConfig(
        learning_rate=0.02,
        max_power=12,
        evaluation_points=512,
        seed=0,
        batch_size=256,
    ),
)
```

Accelerated-path assumptions:

- training uses mini-batch updates
- milestone `updates` still mean optimizer steps, not individual samples
- progress logs include batch size and total samples seen
- omitting `AcceleratedTrainingConfig.runtime` makes training inherit `network.runtime`; setting it explicitly overrides the network for that fit call
- `fit_function_accelerated(...)` requires a vectorized target function that accepts NumPy inputs and returns one scalar output per sample
- `output_modifier` is inference-only; accelerated training and `train_batch(...)` still operate on the raw numeric outputs

## Dataset Training

Reference dataset training:

```python
import numpy as np

from neural_net import TrainingConfig, build_random_network, fit_dataset, predict_dataset
from neural_net.backend import ActivationFunc

x_train = np.array([[0.0], [0.5], [1.0], [1.5]], dtype=float)
y_train = np.sin(x_train)

network = build_random_network(
    input_layer_dim=1,
    hidden_layer_shapes=(16, 16, 1),
    activation=ActivationFunc.tanh,
    seed=0,
)

result = fit_dataset(
    network,
    x_train,
    y_train,
    config=TrainingConfig(max_power=10, seed=0),
)

predictions = predict_dataset(network, x_train)
```

Accelerated dataset training:

```python
import numpy as np

from neural_net import (
    AcceleratedTrainingConfig,
    ActivationFunc,
    build_accelerated_network,
    fit_dataset_accelerated,
    predict_dataset_accelerated,
)

x_train = np.linspace(-np.pi, np.pi, 2048, dtype=float).reshape(-1, 1)
y_train = np.sin(x_train)

network = build_accelerated_network(
    input_layer_dim=1,
    hidden_layer_shapes=(32, 32, 1),
    activation=ActivationFunc.tanh,
    seed=0,
)

result = fit_dataset_accelerated(
    network,
    x_train,
    y_train,
    config=AcceleratedTrainingConfig(max_power=10, batch_size=256, seed=0),
)

predictions = predict_dataset_accelerated(network, x_train)
```

## Benchmarking

Run the included benchmark to compare scalar and accelerated throughput:

```bash
uv run python benchmarks/compare_backends.py
```

The benchmark prints training and inference times plus the measured speedup. By default it treats `10x` as the minimum acceptable acceleration target.

## Tests

Run the test suite with the standard library runner:

```bash
uv run python -m unittest discover -s tests -v
```
