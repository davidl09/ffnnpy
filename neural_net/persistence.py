from __future__ import annotations

import json
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .accelerated import AcceleratedFFNN, AcceleratedRuntime, AcceleratedTrainingConfig
from .backend import ActivationFunc, FFNN, FFNNConfig, LossFunc
from .training import TrainingConfig


FORMAT_VERSION = 1
BACKEND_REFERENCE = "reference"
BACKEND_ACCELERATED = "accelerated"
_METADATA_KEY = "__ffnnpy_metadata__"
_OUTPUT_MODIFIER_REGISTRY: dict[str, Callable[[np.ndarray], Any]] = {}


@dataclass(frozen=True)
class LoadedNetworkArtifact:
    network: FFNN | AcceleratedFFNN
    training_config: TrainingConfig | AcceleratedTrainingConfig | None
    backend: str
    output_modifier_name: str | None
    format_version: int


def register_output_modifier(name: str, modifier: Callable[[np.ndarray], Any]) -> None:
    if not isinstance(name, str) or not name:
        raise ValueError("output modifier name must be a non-empty string")
    if not callable(modifier):
        raise TypeError("output modifier must be callable")

    existing_modifier = _OUTPUT_MODIFIER_REGISTRY.get(name)
    if existing_modifier is not None and existing_modifier is not modifier:
        raise ValueError(f"output modifier name '{name}' is already registered")

    for existing_name, existing_modifier in _OUTPUT_MODIFIER_REGISTRY.items():
        if existing_modifier is modifier and existing_name != name:
            raise ValueError(
                f"output modifier is already registered as '{existing_name}'"
            )

    _OUTPUT_MODIFIER_REGISTRY[name] = modifier


def save_network(
    network: FFNN | AcceleratedFFNN,
    path: str | PathLike[str],
    *,
    training_config: TrainingConfig | AcceleratedTrainingConfig | None = None,
) -> None:
    output_path = Path(path)
    _validate_save_path(output_path)

    backend = _network_backend(network)
    _validate_training_config(training_config, backend=backend)

    metadata = {
        "format_version": FORMAT_VERSION,
        "backend": backend,
        "network": _serialize_network_config(network),
        "output_modifier_name": _resolve_output_modifier_name(network.output_modifier),
        "training_config": _serialize_training_config(training_config, backend=backend),
    }
    archive_arrays = {
        _METADATA_KEY: _metadata_to_array(metadata),
        **_serialize_parameter_arrays(network),
    }

    with output_path.open("wb") as handle:
        np.savez_compressed(handle, **archive_arrays)


def load_network(path: str | PathLike[str]) -> LoadedNetworkArtifact:
    input_path = Path(path)

    try:
        with input_path.open("rb") as handle, np.load(handle, allow_pickle=False) as archive:
            metadata = _load_metadata(archive)
            backend = _validate_metadata(metadata)
            config = _deserialize_network_config(metadata["network"])
            output_modifier_name = metadata["output_modifier_name"]
            output_modifier = _resolve_output_modifier(output_modifier_name)

            if backend == BACKEND_REFERENCE:
                network = FFNN(config)
                _restore_reference_network(
                    network,
                    archive,
                    hidden_layer_shapes=tuple(int(size) for size in config.hidden_layer_shapes),
                )
            else:
                runtime = AcceleratedRuntime(metadata["network"]["runtime"])
                network = AcceleratedFFNN(config, runtime=runtime)
                _restore_accelerated_network(
                    network,
                    archive,
                    hidden_layer_shapes=tuple(int(size) for size in config.hidden_layer_shapes),
                )

            network.output_modifier = output_modifier
            network.config.output_modifier = output_modifier
            training_config = _deserialize_training_config(metadata["training_config"])
    except FileNotFoundError:
        raise
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Invalid FFNNPY file '{input_path}': {exc}") from exc

    return LoadedNetworkArtifact(
        network=network,
        training_config=training_config,
        backend=backend,
        output_modifier_name=output_modifier_name,
        format_version=metadata["format_version"],
    )


def _validate_save_path(path: Path) -> None:
    if path.suffix != ".ffnnpy":
        raise ValueError("save path must end with .ffnnpy")


def _network_backend(network: FFNN | AcceleratedFFNN) -> str:
    if isinstance(network, AcceleratedFFNN):
        return BACKEND_ACCELERATED
    if isinstance(network, FFNN):
        return BACKEND_REFERENCE
    raise TypeError("network must be an FFNN or AcceleratedFFNN instance")


def _validate_training_config(
    training_config: TrainingConfig | AcceleratedTrainingConfig | None,
    *,
    backend: str,
) -> None:
    if training_config is None:
        return

    if backend == BACKEND_REFERENCE and not isinstance(training_config, TrainingConfig):
        raise TypeError("reference networks require a TrainingConfig training_config")
    if backend == BACKEND_ACCELERATED and not isinstance(
        training_config, AcceleratedTrainingConfig
    ):
        raise TypeError(
            "accelerated networks require an AcceleratedTrainingConfig training_config"
        )


def _serialize_network_config(network: FFNN | AcceleratedFFNN) -> dict[str, Any]:
    loss_func = network.config.loss_func
    if not isinstance(loss_func, LossFunc):
        loss_func = LossFunc(loss_func)

    metadata: dict[str, Any] = {
        "input_layer_dim": int(network.config.input_layer_dim),
        "hidden_layer_shapes": [
            int(size) for size in np.asarray(network.config.hidden_layer_shapes, dtype=int)
        ],
        "layer_activation_funcs": [
            activation.value for activation in network.config.layer_activation_funcs
        ],
        "loss_func": loss_func.value,
    }

    if isinstance(network, AcceleratedFFNN):
        metadata["runtime"] = network.runtime.value

    return metadata


def _serialize_parameter_arrays(network: FFNN | AcceleratedFFNN) -> dict[str, np.ndarray]:
    archive_arrays: dict[str, np.ndarray] = {}

    for layer_index, weights in enumerate(network.weights):
        archive_arrays[f"weights_{layer_index}"] = np.array(weights, copy=True)
    for layer_index, biases in enumerate(network.biases):
        archive_arrays[f"biases_{layer_index}"] = np.array(biases, copy=True)

    return archive_arrays


def _resolve_output_modifier_name(
    output_modifier: Callable[[np.ndarray], Any] | None,
) -> str | None:
    if output_modifier is None:
        return None

    for name, registered_modifier in _OUTPUT_MODIFIER_REGISTRY.items():
        if registered_modifier is output_modifier:
            return name

    raise ValueError(
        "output_modifier must be registered with register_output_modifier(...) "
        "before saving"
    )


def _serialize_training_config(
    training_config: TrainingConfig | AcceleratedTrainingConfig | None,
    *,
    backend: str,
) -> dict[str, Any] | None:
    if training_config is None:
        return None

    payload: dict[str, Any] = {
        "backend": backend,
        "learning_rate": float(training_config.learning_rate),
        "max_power": int(training_config.max_power),
        "evaluation_points": int(training_config.evaluation_points),
        "seed": int(training_config.seed),
    }

    if isinstance(training_config, AcceleratedTrainingConfig):
        payload["batch_size"] = int(training_config.batch_size)
        payload["runtime"] = training_config.runtime.value

    return payload


def _deserialize_network_config(metadata: dict[str, Any]) -> FFNNConfig:
    if not isinstance(metadata, dict):
        raise ValueError("network metadata must be a JSON object")

    try:
        input_layer_dim = int(metadata["input_layer_dim"])
        hidden_layer_shapes = tuple(int(size) for size in metadata["hidden_layer_shapes"])
        layer_activation_funcs = tuple(
            ActivationFunc(name) for name in metadata["layer_activation_funcs"]
        )
        loss_func = LossFunc(metadata["loss_func"])
    except KeyError as exc:
        raise ValueError(f"network metadata is missing '{exc.args[0]}'") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(f"network metadata is invalid: {exc}") from exc

    if input_layer_dim < 1:
        raise ValueError("input_layer_dim must be at least 1")
    if not hidden_layer_shapes:
        raise ValueError("hidden_layer_shapes must contain at least one layer")
    if any(size < 1 for size in hidden_layer_shapes):
        raise ValueError("hidden_layer_shapes entries must be at least 1")
    if len(layer_activation_funcs) != len(hidden_layer_shapes):
        raise ValueError(
            "layer_activation_funcs length must match hidden_layer_shapes length"
        )

    return FFNNConfig(
        input_layer_dim=input_layer_dim,
        hidden_layer_count=len(hidden_layer_shapes),
        hidden_layer_shapes=hidden_layer_shapes,
        activation_func=layer_activation_funcs,
        loss_func=loss_func,
    )


def _deserialize_training_config(
    metadata: dict[str, Any] | None,
) -> TrainingConfig | AcceleratedTrainingConfig | None:
    if metadata is None:
        return None
    if not isinstance(metadata, dict):
        raise ValueError("training_config metadata must be a JSON object")

    try:
        backend = metadata["backend"]
        if backend == BACKEND_REFERENCE:
            return TrainingConfig(
                learning_rate=float(metadata["learning_rate"]),
                max_power=int(metadata["max_power"]),
                evaluation_points=int(metadata["evaluation_points"]),
                seed=int(metadata["seed"]),
            )
        if backend == BACKEND_ACCELERATED:
            return AcceleratedTrainingConfig(
                learning_rate=float(metadata["learning_rate"]),
                max_power=int(metadata["max_power"]),
                evaluation_points=int(metadata["evaluation_points"]),
                seed=int(metadata["seed"]),
                batch_size=int(metadata["batch_size"]),
                runtime=AcceleratedRuntime(metadata["runtime"]),
            )
    except KeyError as exc:
        raise ValueError(f"training_config metadata is missing '{exc.args[0]}'") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(f"training_config metadata is invalid: {exc}") from exc

    raise ValueError(f"Unsupported training_config backend '{backend}'")


def _metadata_to_array(metadata: dict[str, Any]) -> np.ndarray:
    metadata_bytes = json.dumps(
        metadata,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return np.frombuffer(metadata_bytes, dtype=np.uint8).copy()


def _load_metadata(archive) -> dict[str, Any]:
    if _METADATA_KEY not in archive:
        raise ValueError("missing FFNNPY metadata entry")

    metadata_array = np.asarray(archive[_METADATA_KEY], dtype=np.uint8)
    try:
        metadata = json.loads(metadata_array.tobytes().decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ValueError("metadata is not valid UTF-8") from exc
    except json.JSONDecodeError as exc:
        raise ValueError("metadata is not valid JSON") from exc

    if not isinstance(metadata, dict):
        raise ValueError("metadata must decode to a JSON object")

    return metadata


def _validate_metadata(metadata: dict[str, Any]) -> str:
    format_version = metadata.get("format_version")
    if format_version != FORMAT_VERSION:
        raise ValueError(f"Unsupported FFNNPY format version: {format_version}")

    backend = metadata.get("backend")
    if backend not in {BACKEND_REFERENCE, BACKEND_ACCELERATED}:
        raise ValueError(f"Unsupported backend '{backend}'")

    output_modifier_name = metadata.get("output_modifier_name")
    if output_modifier_name is not None and not isinstance(output_modifier_name, str):
        raise ValueError("output_modifier_name must be a string or null")

    if "network" not in metadata:
        raise ValueError("metadata is missing 'network'")

    if "training_config" not in metadata:
        raise ValueError("metadata is missing 'training_config'")

    return backend


def _resolve_output_modifier(
    output_modifier_name: str | None,
) -> Callable[[np.ndarray], Any] | None:
    if output_modifier_name is None:
        return None

    output_modifier = _OUTPUT_MODIFIER_REGISTRY.get(output_modifier_name)
    if output_modifier is None:
        raise ValueError(
            f"output modifier '{output_modifier_name}' is not registered in this process"
        )
    return output_modifier


def _restore_reference_network(
    network: FFNN,
    archive,
    *,
    hidden_layer_shapes: tuple[int, ...],
) -> None:
    input_dim = network.config.input_layer_dim

    for layer_index, output_dim in enumerate(hidden_layer_shapes):
        expected_weight_shape = (output_dim, input_dim)
        expected_bias_shape = (output_dim,)
        network.weights[layer_index] = _load_parameter_array(
            archive,
            key=f"weights_{layer_index}",
            expected_shape=expected_weight_shape,
        )
        network.biases[layer_index] = _load_parameter_array(
            archive,
            key=f"biases_{layer_index}",
            expected_shape=expected_bias_shape,
        )
        network.values[layer_index] = np.zeros(output_dim, dtype=float)
        network.pre_activations[layer_index] = np.zeros(output_dim, dtype=float)
        input_dim = output_dim


def _restore_accelerated_network(
    network: AcceleratedFFNN,
    archive,
    *,
    hidden_layer_shapes: tuple[int, ...],
) -> None:
    input_dim = network.config.input_layer_dim

    for layer_index, output_dim in enumerate(hidden_layer_shapes):
        expected_weight_shape = (output_dim, input_dim)
        expected_bias_shape = (output_dim,)
        network.weights[layer_index] = np.ascontiguousarray(
            _load_parameter_array(
                archive,
                key=f"weights_{layer_index}",
                expected_shape=expected_weight_shape,
            )
        )
        network.biases[layer_index] = np.ascontiguousarray(
            _load_parameter_array(
                archive,
                key=f"biases_{layer_index}",
                expected_shape=expected_bias_shape,
            )
        )
        network.values[layer_index] = np.zeros((1, output_dim), dtype=np.float64)
        network.pre_activations[layer_index] = np.zeros((1, output_dim), dtype=np.float64)
        input_dim = output_dim

    network._numba_weights = None
    network._numba_biases = None


def _load_parameter_array(
    archive,
    *,
    key: str,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    if key not in archive:
        raise ValueError(f"missing parameter array '{key}'")

    array = np.array(archive[key], copy=True)
    if array.shape != expected_shape:
        raise ValueError(
            f"parameter array '{key}' has shape {array.shape}, expected {expected_shape}"
        )
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"parameter array '{key}' must be numeric")

    return array


__all__ = [
    "LoadedNetworkArtifact",
    "load_network",
    "register_output_modifier",
    "save_network",
]
