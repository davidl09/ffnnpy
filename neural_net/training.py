from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from queue import SimpleQueue
from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np

from .backend import (
    ActivationFunc,
    FFNN,
    FFNNConfig,
    LossFunc,
    _apply_output_modifier_batch,
    get_loss_func,
)

if TYPE_CHECKING:
    from .accelerated import AcceleratedFFNN


DEFAULT_DOMAIN = (-np.pi, np.pi)
_REFERENCE_TARGET_FUNC_ERROR = (
    "fit_function requires target_func to accept either a float and return a scalar, "
    "or a NumPy batch input and return one scalar output per sample"
)


def powers_of_two_milestones(max_power: int) -> tuple[int, ...]:
    if max_power < 0:
        raise ValueError("max_power must be non-negative")
    return tuple(2**power for power in range(max_power + 1))


DEFAULT_MILESTONES = powers_of_two_milestones(12)


def _normalize_milestones(values: Sequence[int]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError("milestones must be a sequence of positive integers")

    milestones = tuple(int(value) for value in values)
    if not milestones:
        raise ValueError("milestones must contain at least one value")

    previous = 0
    for milestone in milestones:
        if milestone < 1:
            raise ValueError("milestones must contain only positive integers")
        if milestone <= previous:
            raise ValueError("milestones must be strictly increasing")
        previous = milestone

    return milestones


class AsyncProgressPrinter:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self._queue: SimpleQueue[str | None] = SimpleQueue()
        self._thread: threading.Thread | None = None

    def __enter__(self):
        if self.enabled:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def log(self, message: str):
        if self.enabled:
            self._queue.put(message)

    def close(self):
        if self._thread is not None:
            self._queue.put(None)
            self._thread.join()
            self._thread = None

    def _run(self):
        while True:
            message = self._queue.get()
            if message is None:
                break
            print(message, file=sys.stdout, flush=False)
        sys.stdout.flush()


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 0.02
    milestones: tuple[int, ...] = DEFAULT_MILESTONES
    evaluation_points: int = 512
    seed: int = 0

    def __post_init__(self):
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.evaluation_points < 2:
            raise ValueError("evaluation_points must be at least 2")
        object.__setattr__(self, "milestones", _normalize_milestones(self.milestones))


@dataclass
class TrainingResult:
    evaluation_inputs: np.ndarray
    evaluation_targets: np.ndarray
    snapshots: dict[int, np.ndarray]
    losses: dict[int, float]
    network: FFNN | AcceleratedFFNN
    milestones: tuple[int, ...]


def build_random_network(
    *,
    input_layer_dim: int = 1,
    hidden_layer_shapes: tuple[int, ...] = (32, 32, 1),
    activation: ActivationFunc | str | Sequence[ActivationFunc | str] = ActivationFunc.tanh,
    loss_func: LossFunc | str = LossFunc.mse,
    positive_class_weight: float = 1.0,
    seed: int = 0,
    output_modifier: Callable[[np.ndarray], Any] | None = None,
) -> FFNN:
    config = FFNNConfig(
        input_layer_dim=input_layer_dim,
        hidden_layer_count=len(hidden_layer_shapes),
        hidden_layer_shapes=hidden_layer_shapes,
        activation_func=activation,
        loss_func=loss_func,
        positive_class_weight=positive_class_weight,
        output_modifier=output_modifier,
    )
    network = FFNN(config)

    rng = np.random.default_rng(seed)
    input_dim = config.input_layer_dim
    for layer_index, output_dim in enumerate(config.hidden_layer_shapes):
        # Xavier-style scaling keeps tanh-like activations from saturating immediately.
        scale = np.sqrt(2.0 / (input_dim + output_dim))
        network.weights[layer_index] = rng.normal(
            loc=0.0,
            scale=scale,
            size=(output_dim, input_dim),
        )
        network.biases[layer_index] = np.zeros(output_dim, dtype=float)
        network.values[layer_index] = np.zeros(output_dim, dtype=float)
        network.pre_activations[layer_index] = np.zeros(output_dim, dtype=float)
        input_dim = output_dim

    return network


def predict_dataset(network: FFNN, inputs: np.ndarray) -> np.ndarray:
    raw_predictions = _predict_dataset_raw(network, inputs)
    return _apply_output_modifier_batch(raw_predictions, network.output_modifier)


def _predict_dataset_raw(network: FFNN, inputs: np.ndarray) -> np.ndarray:
    input_rows = _normalize_samples(
        inputs,
        feature_dim=network.config.input_layer_dim,
        name="inputs",
    )
    return np.array(
        [network._raw_forward_pass(sample) for sample in input_rows],
        dtype=float,
    )


def fit_dataset(
    network: FFNN,
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    *,
    config: TrainingConfig = TrainingConfig(),
    evaluation_inputs: np.ndarray | None = None,
    evaluation_targets: np.ndarray | None = None,
    progress_logger: Callable[[str], None] | None = None,
) -> TrainingResult:
    output_dim = int(network.config.hidden_layer_shapes[-1])
    train_inputs = _normalize_samples(
        train_inputs,
        feature_dim=network.config.input_layer_dim,
        name="train_inputs",
    )
    train_targets = _normalize_samples(
        train_targets,
        feature_dim=output_dim,
        name="train_targets",
    )

    if train_inputs.shape[0] != train_targets.shape[0]:
        raise ValueError("train_inputs and train_targets must have the same number of samples")
    if train_inputs.shape[0] == 0:
        raise ValueError("train_inputs and train_targets must contain at least one sample")

    if evaluation_inputs is None:
        evaluation_inputs = train_inputs
    if evaluation_targets is None:
        evaluation_targets = train_targets

    evaluation_inputs = _normalize_samples(
        evaluation_inputs,
        feature_dim=network.config.input_layer_dim,
        name="evaluation_inputs",
    )
    evaluation_targets = _normalize_samples(
        evaluation_targets,
        feature_dim=output_dim,
        name="evaluation_targets",
    )

    if evaluation_inputs.shape[0] != evaluation_targets.shape[0]:
        raise ValueError("evaluation_inputs and evaluation_targets must have the same number of samples")
    if evaluation_inputs.shape[0] == 0:
        raise ValueError("evaluation_inputs and evaluation_targets must contain at least one sample")

    rng = np.random.default_rng(config.seed + 1)
    milestones = config.milestones
    snapshots: dict[int, np.ndarray] = {}
    losses: dict[int, float] = {}
    loss_fn = get_loss_func(
        network.config.loss_func,
        positive_class_weight=network.config.positive_class_weight,
    )
    training_start = time.perf_counter()

    if progress_logger is not None:
        progress_logger(
            "training start: "
            f"samples={train_inputs.shape[0]} "
            f"target_samples={milestones[-1]} "
            f"lr={config.learning_rate:.4f} "
            f"shape={_network_shape_text(network)}"
        )

    samples_seen = 0
    for milestone in milestones:
        while samples_seen < milestone:
            sample_index = int(rng.integers(train_inputs.shape[0]))
            x_sample = train_inputs[sample_index]
            y_sample = train_targets[sample_index]
            network.fast_backward_pass(x_sample, y_sample, config.learning_rate)
            samples_seen += 1

        prediction = _predict_dataset_raw(network, evaluation_inputs)
        snapshots[milestone] = prediction
        losses[milestone] = float(loss_fn(evaluation_targets, prediction))
        _log_milestone(
            progress_logger,
            milestone=milestone,
            total_milestone=milestones[-1],
            loss_value=losses[milestone],
            x_sample=x_sample,
            y_sample=y_sample,
            started_at=training_start,
        )

    return TrainingResult(
        evaluation_inputs=evaluation_inputs,
        evaluation_targets=evaluation_targets,
        snapshots=snapshots,
        losses=losses,
        network=network,
        milestones=milestones,
    )


def fit_function(
    network: FFNN,
    target_func,
    *,
    domain: tuple[float, float] = DEFAULT_DOMAIN,
    config: TrainingConfig = TrainingConfig(),
    progress_logger: Callable[[str], None] | None = None,
) -> TrainingResult:
    if network.config.input_layer_dim != 1:
        raise ValueError("fit_function currently supports only scalar inputs")

    output_dim = int(network.config.hidden_layer_shapes[-1])
    if output_dim != 1:
        raise ValueError("fit_function currently supports only scalar outputs")

    rng = np.random.default_rng(config.seed + 1)
    milestones = config.milestones
    evaluation_inputs = np.linspace(
        domain[0],
        domain[1],
        config.evaluation_points,
        dtype=float,
    ).reshape(-1, 1)
    evaluation_targets = _evaluate_reference_target(target_func, evaluation_inputs[:, 0])

    snapshots: dict[int, np.ndarray] = {}
    losses: dict[int, float] = {}
    loss_fn = get_loss_func(
        network.config.loss_func,
        positive_class_weight=network.config.positive_class_weight,
    )
    training_start = time.perf_counter()

    if progress_logger is not None:
        progress_logger(
            "training start: "
            f"domain=[{domain[0]:.3f}, {domain[1]:.3f}] "
            f"target_samples={milestones[-1]} "
            f"lr={config.learning_rate:.4f} "
            f"shape={_network_shape_text(network)}"
        )

    samples_seen = 0
    for milestone in milestones:
        while samples_seen < milestone:
            x_sample = rng.uniform(domain[0], domain[1])
            y_sample = float(
                _evaluate_reference_target(
                    target_func,
                    np.array([x_sample], dtype=float),
                )[0, 0]
            )
            network.fast_backward_pass(
                np.array([x_sample], dtype=float),
                np.array([y_sample], dtype=float),
                config.learning_rate,
            )
            samples_seen += 1

        prediction = _predict_dataset_raw(network, evaluation_inputs)
        snapshots[milestone] = prediction
        losses[milestone] = float(loss_fn(evaluation_targets, prediction))
        _log_milestone(
            progress_logger,
            milestone=milestone,
            total_milestone=milestones[-1],
            loss_value=losses[milestone],
            x_sample=np.array([x_sample], dtype=float),
            y_sample=np.array([y_sample], dtype=float),
            started_at=training_start,
        )

    return TrainingResult(
        evaluation_inputs=evaluation_inputs,
        evaluation_targets=evaluation_targets,
        snapshots=snapshots,
        losses=losses,
        network=network,
        milestones=milestones,
    )


def _normalize_samples(values: np.ndarray, *, feature_dim: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        if feature_dim == 1:
            array = array.reshape(-1, 1)
        elif array.size == feature_dim:
            array = array.reshape(1, feature_dim)
        else:
            raise ValueError(f"{name} must have shape (n_samples, {feature_dim})")
    elif array.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D array")

    if array.shape[1] != feature_dim:
        raise ValueError(f"{name} must have shape (n_samples, {feature_dim})")

    return array


def _evaluate_reference_target(target_func, inputs: np.ndarray) -> np.ndarray:
    sample_inputs = np.asarray(inputs, dtype=float).reshape(-1)

    try:
        vectorized_targets = _normalize_samples(
            np.asarray(target_func(sample_inputs), dtype=float),
            feature_dim=1,
            name="target_func(inputs)",
        )
    except Exception:
        vectorized_targets = None
    else:
        if vectorized_targets.shape[0] == sample_inputs.shape[0]:
            return vectorized_targets.astype(float, copy=False)

    scalar_targets: list[float] = []
    try:
        for sample in sample_inputs:
            scalar_targets.append(
                _coerce_scalar_target_output(target_func(float(sample)))
            )
    except Exception as exc:
        raise ValueError(_REFERENCE_TARGET_FUNC_ERROR) from exc

    return np.asarray(scalar_targets, dtype=float).reshape(-1, 1)


def _coerce_scalar_target_output(raw_output) -> float:
    output = np.asarray(raw_output, dtype=float)
    if output.ndim == 0:
        return float(output)

    flat_output = output.reshape(-1)
    if flat_output.size != 1:
        raise ValueError(_REFERENCE_TARGET_FUNC_ERROR)

    return float(flat_output[0])


def _network_shape_text(network: FFNN) -> str:
    return " -> ".join(
        str(size) for size in (network.config.input_layer_dim, *network.config.hidden_layer_shapes)
    )


def _log_milestone(
    progress_logger: Callable[[str], None] | None,
    *,
    milestone: int,
    total_milestone: int,
    loss_value: float,
    x_sample: np.ndarray,
    y_sample: np.ndarray,
    started_at: float,
    batch_size: int | None = None,
):
    if progress_logger is None:
        return

    elapsed = time.perf_counter() - started_at
    samples_per_second = milestone / elapsed if elapsed > 0 else float("inf")
    batch_text = ""
    if batch_size is not None:
        batch_text = f" batch={batch_size:4d}"

    progress_logger(
        f"samples={milestone:7d}/{total_milestone:7d} "
        f"progress={100 * milestone / total_milestone:6.2f}% "
        f"eval_loss={loss_value:.6f} "
        f"last_x={np.asarray(x_sample).reshape(-1)[0]:+.3f} "
        f"last_y={np.asarray(y_sample).reshape(-1)[0]:+.3f} "
        f"elapsed={elapsed:7.2f}s "
        f"rate={samples_per_second:8.1f} samples/s"
        f"{batch_text}"
    )


__all__ = [
    "DEFAULT_DOMAIN",
    "DEFAULT_MILESTONES",
    "AsyncProgressPrinter",
    "TrainingConfig",
    "TrainingResult",
    "build_random_network",
    "predict_dataset",
    "powers_of_two_milestones",
    "fit_dataset",
    "fit_function",
]
