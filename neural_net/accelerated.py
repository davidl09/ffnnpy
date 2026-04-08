from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Sequence
import time

import numpy as np

from .backend import (
    ActivationFunc,
    FFNNConfig,
    LossFunc,
    _apply_output_modifier,
    _apply_output_modifier_batch,
    get_loss_func,
)
from .training import TrainingResult, _log_milestone, _network_shape_text, _normalize_samples

try:
    from numba import njit
    from numba.typed import List as NumbaList
except ModuleNotFoundError:
    njit = None
    NumbaList = None


_ACTIVATION_RELU = 0
_ACTIVATION_TANH = 1
_ACTIVATION_SIGMOID = 2
_ACTIVATION_INV_QUAD = 3
_LOSS_MSE = 0
_LOSS_CROSS_ENTROPY = 1


class AcceleratedRuntime(Enum):
    auto = "auto"
    numpy = "numpy"
    numba = "numba"


def _coerce_runtime(runtime: AcceleratedRuntime | str) -> AcceleratedRuntime:
    if isinstance(runtime, AcceleratedRuntime):
        return runtime
    return AcceleratedRuntime(runtime)


def _activation_code(activation: ActivationFunc) -> int:
    if activation is ActivationFunc.relu:
        return _ACTIVATION_RELU
    if activation is ActivationFunc.tanh:
        return _ACTIVATION_TANH
    if activation is ActivationFunc.sigmoid:
        return _ACTIVATION_SIGMOID
    if activation is ActivationFunc.inv_quad:
        return _ACTIVATION_INV_QUAD
    raise ValueError(f"Activation {activation} not supported")


def _loss_code(loss_func: LossFunc) -> int:
    if loss_func is LossFunc.mse:
        return _LOSS_MSE
    if loss_func is LossFunc.cross_entropy:
        return _LOSS_CROSS_ENTROPY
    raise ValueError(f"Loss {loss_func} not supported")


@dataclass(frozen=True)
class AcceleratedTrainingConfig:
    learning_rate: float = 0.02
    max_power: int = 12
    evaluation_points: int = 512
    seed: int = 0
    batch_size: int = 256
    runtime: AcceleratedRuntime | None = None

    def __post_init__(self):
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.max_power < 0:
            raise ValueError("max_power must be non-negative")
        if self.evaluation_points < 2:
            raise ValueError("evaluation_points must be at least 2")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        if self.runtime is not None:
            object.__setattr__(self, "runtime", _coerce_runtime(self.runtime))

    @property
    def milestone_steps(self) -> tuple[int, ...]:
        return tuple(2**power for power in range(self.max_power + 1))


class AcceleratedFFNN:
    def __init__(
        self,
        config: FFNNConfig,
        *,
        runtime: AcceleratedRuntime = AcceleratedRuntime.auto,
    ):
        self.config = config
        self.output_modifier = self.config.output_modifier
        self.runtime = _coerce_runtime(runtime)
        self.activation_codes = np.ascontiguousarray(
            np.array(
                [_activation_code(func) for func in self.config.layer_activation_funcs],
                dtype=np.int64,
            )
        )
        self.loss_code = _loss_code(self.config.loss_func)

        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        self.pre_activations: list[np.ndarray] = []
        self.values: list[np.ndarray] = []

        input_dim = self.config.input_layer_dim
        for output_dim in self.config.hidden_layer_shapes:
            output_dim = int(output_dim)
            self.weights.append(np.random.random((output_dim, input_dim)).astype(np.float64))
            self.biases.append(np.random.random(output_dim).astype(np.float64))
            self.pre_activations.append(np.zeros((1, output_dim), dtype=np.float64))
            self.values.append(np.zeros((1, output_dim), dtype=np.float64))
            input_dim = output_dim

        self._numba_weights = None
        self._numba_biases = None

    def _forward_batch_raw(
        self,
        inputs: np.ndarray | list[float] | tuple[float, ...],
        *,
        runtime: AcceleratedRuntime | str | None = None,
    ) -> np.ndarray:
        input_rows = _normalize_samples(
            inputs,
            feature_dim=self.config.input_layer_dim,
            name="inputs",
        ).astype(np.float64, copy=False)

        resolved_runtime = self.resolve_runtime(runtime)
        if resolved_runtime is AcceleratedRuntime.numba:
            self._ensure_numba_lists()
            outputs = _forward_batch_numba(
                self._numba_weights,
                self._numba_biases,
                np.ascontiguousarray(input_rows),
                self.activation_codes,
            )
        else:
            outputs, pre_activations, values = _forward_batch_numpy(
                self.weights,
                self.biases,
                input_rows,
                self.config.layer_activation_funcs,
                capture_intermediates=True,
            )
            self.pre_activations = pre_activations
            self.values = values
            return outputs

        _, pre_activations, values = _forward_batch_numpy(
            self.weights,
            self.biases,
            input_rows,
            self.config.layer_activation_funcs,
            capture_intermediates=True,
        )
        self.pre_activations = pre_activations
        self.values = values
        return outputs

    def forward_batch(
        self,
        inputs: np.ndarray | list[float] | tuple[float, ...],
        *,
        runtime: AcceleratedRuntime | str | None = None,
    ) -> np.ndarray:
        raw_outputs = self._forward_batch_raw(inputs, runtime=runtime)
        return _apply_output_modifier_batch(raw_outputs, self.output_modifier)

    def fast_forward_pass(self, x_: np.ndarray | list[float] | tuple[float, ...]):
        input_rows = _normalize_samples(
            x_,
            feature_dim=self.config.input_layer_dim,
            name="x_",
        ).astype(np.float64, copy=False)
        if input_rows.shape[0] != 1:
            raise ValueError(
                "fast_forward_pass expects exactly one sample; "
                "use forward_batch(...) or predict_dataset_accelerated(...) for batched inputs"
            )

        raw_output = self._forward_batch_raw(input_rows).reshape(-1)
        return _apply_output_modifier(self.output_modifier, raw_output)

    def train_batch(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        learning_rate: float,
        *,
        runtime: AcceleratedRuntime | str | None = None,
    ) -> np.ndarray:
        input_rows = _normalize_samples(
            inputs,
            feature_dim=self.config.input_layer_dim,
            name="inputs",
        ).astype(np.float64, copy=False)
        output_dim = int(self.config.hidden_layer_shapes[-1])
        target_rows = _normalize_samples(
            targets,
            feature_dim=output_dim,
            name="targets",
        ).astype(np.float64, copy=False)

        if input_rows.shape[0] != target_rows.shape[0]:
            raise ValueError("inputs and targets must have the same number of samples")

        if self.config.loss_func is LossFunc.cross_entropy:
            _validate_cross_entropy_targets_numpy(target_rows)

        resolved_runtime = self.resolve_runtime(runtime)
        if resolved_runtime is AcceleratedRuntime.numba:
            self._ensure_numba_lists()
            predictions = _train_batch_numba(
                self._numba_weights,
                self._numba_biases,
                np.ascontiguousarray(input_rows),
                np.ascontiguousarray(target_rows),
                float(learning_rate),
                self.activation_codes,
                self.loss_code,
                float(self.config.positive_class_weight),
            )
        else:
            predictions, pre_activations, values = _train_batch_numpy(
                self.weights,
                self.biases,
                input_rows,
                target_rows,
                float(learning_rate),
                self.config.layer_activation_funcs,
                self.config.loss_func,
                self.config.positive_class_weight,
            )
            self.pre_activations = pre_activations
            self.values = values
            return predictions

        _, pre_activations, values = _forward_batch_numpy(
            self.weights,
            self.biases,
            input_rows,
            self.config.layer_activation_funcs,
            capture_intermediates=True,
        )
        self.pre_activations = pre_activations
        self.values = values
        return predictions

    def resolve_runtime(
        self,
        runtime: AcceleratedRuntime | str | None = None,
    ) -> AcceleratedRuntime:
        requested = self.runtime if runtime is None else _coerce_runtime(runtime)
        if requested is AcceleratedRuntime.auto:
            return AcceleratedRuntime.numba if njit is not None else AcceleratedRuntime.numpy
        if requested is AcceleratedRuntime.numba and njit is None:
            raise RuntimeError(
                "runtime='numba' requested but numba is not installed. "
                "Install the optional accelerated extra to enable it."
            )
        return requested

    def _ensure_numba_lists(self):
        if njit is None or NumbaList is None:
            raise RuntimeError(
                "runtime='numba' requested but numba is not installed. "
                "Install the optional accelerated extra to enable it."
            )
        if self._numba_weights is None or self._numba_biases is None:
            typed_weights = NumbaList()
            typed_biases = NumbaList()
            for weights in self.weights:
                typed_weights.append(np.ascontiguousarray(weights))
            for biases in self.biases:
                typed_biases.append(np.ascontiguousarray(biases))
            self._numba_weights = typed_weights
            self._numba_biases = typed_biases


def build_accelerated_network(
    *,
    input_layer_dim: int = 1,
    hidden_layer_shapes: tuple[int, ...] = (32, 32, 1),
    activation: ActivationFunc | str | Sequence[ActivationFunc | str] = ActivationFunc.tanh,
    loss_func: LossFunc | str = LossFunc.mse,
    positive_class_weight: float = 1.0,
    seed: int = 0,
    runtime: AcceleratedRuntime = AcceleratedRuntime.auto,
    output_modifier: Callable[[np.ndarray], Any] | None = None,
) -> AcceleratedFFNN:
    config = FFNNConfig(
        input_layer_dim=input_layer_dim,
        hidden_layer_count=len(hidden_layer_shapes),
        hidden_layer_shapes=hidden_layer_shapes,
        activation_func=activation,
        loss_func=loss_func,
        positive_class_weight=positive_class_weight,
        output_modifier=output_modifier,
    )
    network = AcceleratedFFNN(config, runtime=runtime)

    rng = np.random.default_rng(seed)
    input_dim = config.input_layer_dim
    for layer_index, output_dim in enumerate(config.hidden_layer_shapes):
        output_dim = int(output_dim)
        scale = np.sqrt(2.0 / (input_dim + output_dim))
        network.weights[layer_index] = np.ascontiguousarray(
            rng.normal(
                loc=0.0,
                scale=scale,
                size=(output_dim, input_dim),
            ).astype(np.float64)
        )
        network.biases[layer_index] = np.zeros(output_dim, dtype=np.float64)
        network.values[layer_index] = np.zeros((1, output_dim), dtype=np.float64)
        network.pre_activations[layer_index] = np.zeros((1, output_dim), dtype=np.float64)
        input_dim = output_dim

    return network


def predict_dataset_accelerated(
    network: AcceleratedFFNN,
    inputs: np.ndarray,
    *,
    runtime: AcceleratedRuntime | str | None = None,
) -> np.ndarray:
    return network.forward_batch(inputs, runtime=runtime)


def fit_dataset_accelerated(
    network: AcceleratedFFNN,
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    *,
    config: AcceleratedTrainingConfig = AcceleratedTrainingConfig(),
    evaluation_inputs: np.ndarray | None = None,
    evaluation_targets: np.ndarray | None = None,
    progress_logger: Callable[[str], None] | None = None,
) -> TrainingResult:
    output_dim = int(network.config.hidden_layer_shapes[-1])
    train_inputs = _normalize_samples(
        train_inputs,
        feature_dim=network.config.input_layer_dim,
        name="train_inputs",
    ).astype(np.float64, copy=False)
    train_targets = _normalize_samples(
        train_targets,
        feature_dim=output_dim,
        name="train_targets",
    ).astype(np.float64, copy=False)

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
    ).astype(np.float64, copy=False)
    evaluation_targets = _normalize_samples(
        evaluation_targets,
        feature_dim=output_dim,
        name="evaluation_targets",
    ).astype(np.float64, copy=False)

    if evaluation_inputs.shape[0] != evaluation_targets.shape[0]:
        raise ValueError("evaluation_inputs and evaluation_targets must have the same number of samples")
    if evaluation_inputs.shape[0] == 0:
        raise ValueError("evaluation_inputs and evaluation_targets must contain at least one sample")

    rng = np.random.default_rng(config.seed + 1)
    milestones = config.milestone_steps
    milestone_set = set(milestones)
    snapshots: dict[int, np.ndarray] = {}
    losses: dict[int, float] = {}
    loss_fn = get_loss_func(
        network.config.loss_func,
        positive_class_weight=network.config.positive_class_weight,
    )
    training_start = time.perf_counter()
    resolved_runtime = network.resolve_runtime(config.runtime)

    if progress_logger is not None:
        progress_logger(
            "training start: "
            f"samples={train_inputs.shape[0]} "
            f"updates={milestones[-1]} "
            f"batch_size={config.batch_size} "
            f"lr={config.learning_rate:.4f} "
            f"runtime={resolved_runtime.value} "
            f"shape={_network_shape_text(network)}"
        )

    for step in range(1, milestones[-1] + 1):
        sample_index = rng.integers(train_inputs.shape[0], size=config.batch_size)
        x_batch = train_inputs[sample_index]
        y_batch = train_targets[sample_index]
        network.train_batch(x_batch, y_batch, config.learning_rate, runtime=resolved_runtime)

        if step in milestone_set:
            prediction = network._forward_batch_raw(evaluation_inputs, runtime=resolved_runtime)
            snapshots[step] = np.array(prediction, copy=True)
            losses[step] = float(loss_fn(evaluation_targets, prediction))
            _log_milestone(
                progress_logger,
                step=step,
                total_steps=milestones[-1],
                loss_value=losses[step],
                x_sample=x_batch[-1],
                y_sample=y_batch[-1],
                started_at=training_start,
                batch_size=config.batch_size,
                samples_seen=step * config.batch_size,
            )

    return TrainingResult(
        evaluation_inputs=evaluation_inputs,
        evaluation_targets=evaluation_targets,
        snapshots=snapshots,
        losses=losses,
        network=network,
        milestone_steps=milestones,
    )


def fit_function_accelerated(
    network: AcceleratedFFNN,
    target_func,
    *,
    domain: tuple[float, float] = (-np.pi, np.pi),
    config: AcceleratedTrainingConfig = AcceleratedTrainingConfig(),
    progress_logger: Callable[[str], None] | None = None,
) -> TrainingResult:
    if network.config.input_layer_dim != 1:
        raise ValueError("fit_function_accelerated currently supports only scalar inputs")

    output_dim = int(network.config.hidden_layer_shapes[-1])
    if output_dim != 1:
        raise ValueError("fit_function_accelerated currently supports only scalar outputs")

    rng = np.random.default_rng(config.seed + 1)
    milestones = config.milestone_steps
    milestone_set = set(milestones)
    evaluation_inputs = np.linspace(
        domain[0],
        domain[1],
        config.evaluation_points,
        dtype=float,
    ).reshape(-1, 1)
    evaluation_targets = _evaluate_vectorized_target(target_func, evaluation_inputs)

    snapshots: dict[int, np.ndarray] = {}
    losses: dict[int, float] = {}
    loss_fn = get_loss_func(
        network.config.loss_func,
        positive_class_weight=network.config.positive_class_weight,
    )
    training_start = time.perf_counter()
    resolved_runtime = network.resolve_runtime(config.runtime)

    if progress_logger is not None:
        progress_logger(
            "training start: "
            f"domain=[{domain[0]:.3f}, {domain[1]:.3f}] "
            f"updates={milestones[-1]} "
            f"batch_size={config.batch_size} "
            f"lr={config.learning_rate:.4f} "
            f"runtime={resolved_runtime.value} "
            f"shape={_network_shape_text(network)}"
        )

    for step in range(1, milestones[-1] + 1):
        x_batch = rng.uniform(domain[0], domain[1], size=(config.batch_size, 1))
        y_batch = _evaluate_vectorized_target(target_func, x_batch)
        network.train_batch(x_batch, y_batch, config.learning_rate, runtime=resolved_runtime)

        if step in milestone_set:
            prediction = network._forward_batch_raw(evaluation_inputs, runtime=resolved_runtime)
            snapshots[step] = np.array(prediction, copy=True)
            losses[step] = float(loss_fn(evaluation_targets, prediction))
            _log_milestone(
                progress_logger,
                step=step,
                total_steps=milestones[-1],
                loss_value=losses[step],
                x_sample=x_batch[-1],
                y_sample=y_batch[-1],
                started_at=training_start,
                batch_size=config.batch_size,
                samples_seen=step * config.batch_size,
            )

    return TrainingResult(
        evaluation_inputs=evaluation_inputs,
        evaluation_targets=evaluation_targets,
        snapshots=snapshots,
        losses=losses,
        network=network,
        milestone_steps=milestones,
    )


def _activate_numpy(values: np.ndarray, activation: ActivationFunc) -> np.ndarray:
    if activation is ActivationFunc.relu:
        return np.maximum(values, 0.0)
    if activation is ActivationFunc.tanh:
        return np.tanh(values)
    if activation is ActivationFunc.sigmoid:
        return 1.0 / (1.0 + np.exp(-values))
    if activation is ActivationFunc.inv_quad:
        return 1.0 / (1.0 + values**2)
    raise ValueError(f"Activation {activation} not supported")


def _activation_derivative_numpy(
    activation: ActivationFunc,
    activated: np.ndarray,
    pre_activated: np.ndarray,
) -> np.ndarray:
    if activation is ActivationFunc.relu:
        return (pre_activated > 0.0).astype(np.float64)
    if activation is ActivationFunc.tanh:
        return 1.0 - activated * activated
    if activation is ActivationFunc.sigmoid:
        return activated * (1.0 - activated)
    if activation is ActivationFunc.inv_quad:
        return -2.0 * pre_activated / (1.0 + pre_activated**2) ** 2
    raise ValueError(f"Activation {activation} not supported")


def _validate_cross_entropy_targets_numpy(targets: np.ndarray) -> None:
    if np.any(~np.isfinite(targets)) or np.any((targets < 0.0) | (targets > 1.0)):
        raise ValueError("cross_entropy targets must be finite and lie in [0, 1]")


def _validate_cross_entropy_predictions_numpy(predictions: np.ndarray) -> None:
    if np.any(~np.isfinite(predictions)) or np.any((predictions < 0.0) | (predictions > 1.0)):
        raise ValueError("cross_entropy predictions must be finite and lie in [0, 1]")


def _output_delta_numpy(
    loss_func: LossFunc,
    predictions: np.ndarray,
    targets: np.ndarray,
    output_activation: ActivationFunc,
    activated_output: np.ndarray,
    pre_activated_output: np.ndarray,
    positive_class_weight: float,
) -> np.ndarray:
    scale = float(targets.size)
    if loss_func is LossFunc.cross_entropy:
        if output_activation is not ActivationFunc.sigmoid:
            raise ValueError("cross_entropy loss requires a sigmoid output activation")
        _validate_cross_entropy_predictions_numpy(predictions)
        return (
            ((1.0 - targets) * predictions)
            - (positive_class_weight * targets * (1.0 - predictions))
        ) / scale

    return (
        (2.0 * (predictions - targets) / scale)
        * _activation_derivative_numpy(
            output_activation,
            activated_output,
            pre_activated_output,
        )
    )


def _forward_batch_numpy(
    weights: list[np.ndarray],
    biases: list[np.ndarray],
    inputs: np.ndarray,
    layer_activation_funcs: Sequence[ActivationFunc],
    *,
    capture_intermediates: bool,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    current = inputs
    pre_activations: list[np.ndarray] = []
    layer_outputs: list[np.ndarray] = []

    for weights_layer, biases_layer, activation in zip(
        weights,
        biases,
        layer_activation_funcs,
        strict=True,
    ):
        pre_activation = current @ weights_layer.T + biases_layer
        current = _activate_numpy(pre_activation, activation)
        if capture_intermediates:
            pre_activations.append(pre_activation)
            layer_outputs.append(current)

    return current, pre_activations, layer_outputs


def _train_batch_numpy(
    weights: list[np.ndarray],
    biases: list[np.ndarray],
    inputs: np.ndarray,
    targets: np.ndarray,
    learning_rate: float,
    layer_activation_funcs: Sequence[ActivationFunc],
    loss_func: LossFunc,
    positive_class_weight: float,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    layer_inputs: list[np.ndarray] = [inputs]
    pre_activations: list[np.ndarray] = []
    layer_outputs: list[np.ndarray] = []
    current = inputs

    for weights_layer, biases_layer, activation in zip(
        weights,
        biases,
        layer_activation_funcs,
        strict=True,
    ):
        pre_activation = current @ weights_layer.T + biases_layer
        current = _activate_numpy(pre_activation, activation)
        pre_activations.append(pre_activation)
        layer_outputs.append(current)
        layer_inputs.append(current)

    predictions = layer_outputs[-1]
    deltas: list[np.ndarray] = [np.zeros_like(pre_activation) for pre_activation in pre_activations]
    deltas[-1] = _output_delta_numpy(
        loss_func,
        predictions,
        targets,
        layer_activation_funcs[-1],
        layer_outputs[-1],
        pre_activations[-1],
        positive_class_weight,
    )

    for layer_index in range(len(weights) - 2, -1, -1):
        deltas[layer_index] = (
            deltas[layer_index + 1] @ weights[layer_index + 1]
        ) * _activation_derivative_numpy(
            layer_activation_funcs[layer_index],
            layer_outputs[layer_index],
            pre_activations[layer_index],
        )

    for layer_index in range(len(weights)):
        weights[layer_index] -= learning_rate * (deltas[layer_index].T @ layer_inputs[layer_index])
        biases[layer_index] -= learning_rate * deltas[layer_index].sum(axis=0)

    return predictions, pre_activations, layer_outputs


def _evaluate_vectorized_target(target_func, batch_inputs: np.ndarray) -> np.ndarray:
    try:
        raw_targets = np.asarray(target_func(batch_inputs[:, 0]), dtype=float)
    except Exception as exc:  # pragma: no cover - surface the original error context.
        raise ValueError(
            "fit_function_accelerated requires target_func to accept a NumPy batch input "
            "and return one scalar output per sample"
        ) from exc

    try:
        targets = _normalize_samples(
            raw_targets,
            feature_dim=1,
            name="target_func(batch_inputs)",
        ).astype(np.float64, copy=False)
    except ValueError as exc:
        raise ValueError(
            "fit_function_accelerated requires target_func to accept a NumPy batch input "
            "and return one scalar output per sample"
        ) from exc

    if targets.shape[0] != batch_inputs.shape[0]:
        raise ValueError(
            "fit_function_accelerated requires target_func to return one scalar output per input sample"
        )

    return targets


if njit is not None:
    @njit(cache=True)
    def _activate_numba(values: np.ndarray, activation_code: int) -> np.ndarray:
        if activation_code == _ACTIVATION_RELU:
            return np.maximum(values, 0.0)
        if activation_code == _ACTIVATION_TANH:
            return np.tanh(values)
        if activation_code == _ACTIVATION_SIGMOID:
            return 1.0 / (1.0 + np.exp(-values))
        return 1.0 / (1.0 + values * values)


    @njit(cache=True)
    def _activation_derivative_numba(
        activated: np.ndarray,
        pre_activated: np.ndarray,
        activation_code: int,
    ) -> np.ndarray:
        if activation_code == _ACTIVATION_RELU:
            return (pre_activated > 0.0).astype(np.float64)
        if activation_code == _ACTIVATION_TANH:
            return 1.0 - activated * activated
        if activation_code == _ACTIVATION_SIGMOID:
            return activated * (1.0 - activated)
        return -2.0 * pre_activated / (1.0 + pre_activated * pre_activated) ** 2


    @njit(cache=True)
    def _forward_batch_numba(
        weights,
        biases,
        inputs: np.ndarray,
        activation_codes: np.ndarray,
    ) -> np.ndarray:
        current = inputs
        for layer_index in range(len(weights)):
            current = _activate_numba(
                current @ weights[layer_index].T + biases[layer_index],
                activation_codes[layer_index],
            )
        return current


    @njit(cache=True)
    def _train_batch_numba(
        weights,
        biases,
        inputs: np.ndarray,
        targets: np.ndarray,
        learning_rate: float,
        activation_codes: np.ndarray,
        loss_code: int,
        positive_class_weight: float,
    ) -> np.ndarray:
        layer_inputs = NumbaList()
        pre_activations = NumbaList()
        activations = NumbaList()
        current = inputs
        layer_inputs.append(inputs)

        for layer_index in range(len(weights)):
            pre_activation = current @ weights[layer_index].T + biases[layer_index]
            current = _activate_numba(pre_activation, activation_codes[layer_index])
            pre_activations.append(pre_activation)
            activations.append(current)
            layer_inputs.append(current)

        predictions = activations[-1]
        deltas = NumbaList()
        for layer_index in range(len(weights)):
            deltas.append(np.zeros_like(pre_activations[layer_index]))

        if loss_code == _LOSS_CROSS_ENTROPY:
            deltas[-1] = (
                ((1.0 - targets) * predictions)
                - (positive_class_weight * targets * (1.0 - predictions))
            ) / targets.size
        else:
            deltas[-1] = (
                (2.0 * (predictions - targets) / targets.size)
                * _activation_derivative_numba(
                    activations[-1],
                    pre_activations[-1],
                    activation_codes[-1],
                )
            )

        for layer_index in range(len(weights) - 2, -1, -1):
            deltas[layer_index] = (
                deltas[layer_index + 1] @ weights[layer_index + 1]
            ) * _activation_derivative_numba(
                activations[layer_index],
                pre_activations[layer_index],
                activation_codes[layer_index],
            )

        for layer_index in range(len(weights)):
            weights[layer_index] -= learning_rate * (deltas[layer_index].T @ layer_inputs[layer_index])
            biases[layer_index] -= learning_rate * np.sum(deltas[layer_index], axis=0)

        return predictions
else:
    def _forward_batch_numba(*args, **kwargs):  # pragma: no cover - exercised only when numba is missing.
        raise RuntimeError("Numba runtime requested but numba is not installed")


    def _train_batch_numba(*args, **kwargs):  # pragma: no cover - exercised only when numba is missing.
        raise RuntimeError("Numba runtime requested but numba is not installed")


__all__ = [
    "AcceleratedRuntime",
    "AcceleratedTrainingConfig",
    "AcceleratedFFNN",
    "build_accelerated_network",
    "predict_dataset_accelerated",
    "fit_dataset_accelerated",
    "fit_function_accelerated",
]
