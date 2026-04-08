from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from neural_net import (
    AcceleratedRuntime,
    AcceleratedTrainingConfig,
    LoadedNetworkArtifact,
    LossFunc,
    TrainingConfig,
    build_accelerated_network,
    build_random_network,
    load_network,
    predict_dataset,
    predict_dataset_accelerated,
    register_output_modifier,
    save_network,
)


def _boolean_threshold(output: np.ndarray) -> bool:
    return bool(output[0] >= 0.5)


class PersistenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        register_output_modifier("boolean_threshold", _boolean_threshold)

    def test_reference_round_trip_preserves_network_and_training_config(self):
        network = build_random_network(
            input_layer_dim=2,
            hidden_layer_shapes=(4, 3, 1),
            activation=(
                "relu",
                "tanh",
                "sigmoid",
            ),
            loss_func=LossFunc.cross_entropy,
            seed=7,
        )
        training_config = TrainingConfig(
            learning_rate=0.03,
            max_power=8,
            evaluation_points=64,
            seed=11,
        )
        inputs = np.array(
            [
                [-1.0, 0.5],
                [0.0, 0.0],
                [1.5, -0.25],
            ],
            dtype=float,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "reference.ffnnpy"
            save_network(network, path, training_config=training_config)
            artifact = load_network(path)

        self.assertIsInstance(artifact, LoadedNetworkArtifact)
        self.assertEqual(artifact.backend, "reference")
        self.assertEqual(artifact.training_config, training_config)
        self.assertIsNone(artifact.output_modifier_name)
        self.assertEqual(artifact.format_version, 1)

        loaded_network = artifact.network
        self.assertEqual(loaded_network.config.input_layer_dim, network.config.input_layer_dim)
        self.assertEqual(
            tuple(int(size) for size in loaded_network.config.hidden_layer_shapes),
            tuple(int(size) for size in network.config.hidden_layer_shapes),
        )
        self.assertEqual(
            loaded_network.config.layer_activation_funcs,
            network.config.layer_activation_funcs,
        )
        self.assertIs(loaded_network.config.loss_func, LossFunc.cross_entropy)

        for layer_index in range(network.config.hidden_layer_count):
            self.assertTrue(
                np.array_equal(loaded_network.weights[layer_index], network.weights[layer_index])
            )
            self.assertTrue(
                np.array_equal(loaded_network.biases[layer_index], network.biases[layer_index])
            )
            self.assertTrue(
                np.array_equal(
                    np.asarray(loaded_network.values[layer_index]),
                    np.zeros_like(np.asarray(loaded_network.values[layer_index])),
                )
            )
            self.assertTrue(
                np.array_equal(
                    np.asarray(loaded_network.pre_activations[layer_index]),
                    np.zeros_like(np.asarray(loaded_network.pre_activations[layer_index])),
                )
            )

        self.assertTrue(
            np.array_equal(
                predict_dataset(network, inputs),
                predict_dataset(loaded_network, inputs),
            )
        )

    def test_accelerated_round_trip_preserves_network_and_training_config(self):
        network = build_accelerated_network(
            input_layer_dim=2,
            hidden_layer_shapes=(5, 2),
            activation=("tanh", "sigmoid"),
            loss_func=LossFunc.cross_entropy,
            seed=3,
            runtime=AcceleratedRuntime.numpy,
        )
        training_config = AcceleratedTrainingConfig(
            learning_rate=0.04,
            max_power=7,
            evaluation_points=32,
            seed=13,
            batch_size=9,
            runtime=AcceleratedRuntime.numpy,
        )
        inputs = np.array(
            [
                [-0.5, -1.0],
                [0.25, 0.75],
                [1.0, -0.5],
            ],
            dtype=float,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "accelerated.ffnnpy"
            save_network(network, path, training_config=training_config)
            artifact = load_network(path)

        self.assertEqual(artifact.backend, "accelerated")
        self.assertEqual(artifact.training_config, training_config)
        self.assertIsNone(artifact.output_modifier_name)

        loaded_network = artifact.network
        self.assertEqual(loaded_network.runtime, AcceleratedRuntime.numpy)
        self.assertIs(loaded_network.config.loss_func, LossFunc.cross_entropy)
        self.assertIsNone(loaded_network._numba_weights)
        self.assertIsNone(loaded_network._numba_biases)

        for layer_index in range(len(network.weights)):
            self.assertTrue(
                np.array_equal(loaded_network.weights[layer_index], network.weights[layer_index])
            )
            self.assertTrue(
                np.array_equal(loaded_network.biases[layer_index], network.biases[layer_index])
            )
            self.assertTrue(
                np.array_equal(
                    loaded_network.values[layer_index],
                    np.zeros_like(loaded_network.values[layer_index]),
                )
            )
            self.assertTrue(
                np.array_equal(
                    loaded_network.pre_activations[layer_index],
                    np.zeros_like(loaded_network.pre_activations[layer_index]),
                )
            )

        self.assertTrue(
            np.array_equal(
                predict_dataset_accelerated(network, inputs, runtime=AcceleratedRuntime.numpy),
                predict_dataset_accelerated(
                    loaded_network,
                    inputs,
                    runtime=AcceleratedRuntime.numpy,
                ),
            )
        )

    def test_accelerated_round_trip_preserves_training_config_without_runtime_override(self):
        network = build_accelerated_network(
            input_layer_dim=2,
            hidden_layer_shapes=(5, 2),
            activation=("tanh", "sigmoid"),
            loss_func=LossFunc.cross_entropy,
            seed=3,
            runtime=AcceleratedRuntime.numpy,
        )
        training_config = AcceleratedTrainingConfig(
            learning_rate=0.04,
            max_power=7,
            evaluation_points=32,
            seed=13,
            batch_size=9,
            runtime=None,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "accelerated-no-runtime.ffnnpy"
            save_network(network, path, training_config=training_config)
            artifact = load_network(path)

        self.assertEqual(artifact.backend, "accelerated")
        self.assertEqual(artifact.training_config, training_config)
        self.assertIsNone(artifact.training_config.runtime)

    def test_registered_output_modifier_round_trips(self):
        network = build_random_network(
            input_layer_dim=1,
            hidden_layer_shapes=(1,),
            activation="sigmoid",
            seed=0,
            output_modifier=_boolean_threshold,
        )
        network.weights[0][:] = 0.0
        network.biases[0][:] = 0.6

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "modifier.ffnnpy"
            save_network(network, path)
            artifact = load_network(path)

        outputs = predict_dataset(artifact.network, np.array([[-1.0], [0.0], [1.0]], dtype=float))

        self.assertEqual(artifact.output_modifier_name, "boolean_threshold")
        self.assertIs(artifact.network.output_modifier, _boolean_threshold)
        self.assertEqual(outputs.dtype, np.bool_)
        self.assertTrue(np.all(outputs))

    def test_unregistered_output_modifier_is_rejected_on_save(self):
        network = build_random_network(
            input_layer_dim=1,
            hidden_layer_shapes=(1,),
            activation="sigmoid",
            seed=0,
            output_modifier=lambda output: bool(output[0] >= 0.5),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad_modifier.ffnnpy"
            with self.assertRaisesRegex(ValueError, "register_output_modifier"):
                save_network(network, path)

    def test_save_rejects_mismatched_training_config(self):
        reference_network = build_random_network(seed=0)
        accelerated_network = build_accelerated_network(seed=0)

        with tempfile.TemporaryDirectory() as temp_dir:
            reference_path = Path(temp_dir) / "reference.ffnnpy"
            accelerated_path = Path(temp_dir) / "accelerated.ffnnpy"

            with self.assertRaisesRegex(TypeError, "TrainingConfig"):
                save_network(
                    reference_network,
                    reference_path,
                    training_config=AcceleratedTrainingConfig(),
                )

            with self.assertRaisesRegex(TypeError, "AcceleratedTrainingConfig"):
                save_network(
                    accelerated_network,
                    accelerated_path,
                    training_config=TrainingConfig(),
                )

    def test_save_requires_ffnnpy_extension(self):
        network = build_random_network(seed=0)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "network.npz"
            with self.assertRaisesRegex(ValueError, r"\.ffnnpy"):
                save_network(network, path)

    def test_load_rejects_malformed_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "malformed.ffnnpy"
            self._write_archive(path, metadata_bytes=b"{not-json")

            with self.assertRaisesRegex(ValueError, "metadata"):
                load_network(path)

    def test_load_rejects_unknown_schema_version(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "unknown_version.ffnnpy"
            self._write_archive(
                path,
                metadata={
                    "format_version": 999,
                    "backend": "reference",
                    "network": {
                        "input_layer_dim": 1,
                        "hidden_layer_shapes": [1],
                        "layer_activation_funcs": ["tanh"],
                        "loss_func": "mse",
                    },
                    "output_modifier_name": None,
                    "training_config": None,
                },
            )

            with self.assertRaisesRegex(ValueError, "Unsupported FFNNPY format version"):
                load_network(path)

    def test_load_rejects_unknown_output_modifier_name(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "unknown_modifier.ffnnpy"
            self._write_archive(
                path,
                metadata={
                    "format_version": 1,
                    "backend": "reference",
                    "network": {
                        "input_layer_dim": 1,
                        "hidden_layer_shapes": [1],
                        "layer_activation_funcs": ["sigmoid"],
                        "loss_func": "mse",
                    },
                    "output_modifier_name": "not_registered",
                    "training_config": None,
                },
                weights_0=np.array([[1.0]], dtype=float),
                biases_0=np.array([0.0], dtype=float),
            )

            with self.assertRaisesRegex(ValueError, "not registered"):
                load_network(path)

    def test_load_rejects_invalid_weight_shape(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad_shapes.ffnnpy"
            self._write_archive(
                path,
                metadata={
                    "format_version": 1,
                    "backend": "reference",
                    "network": {
                        "input_layer_dim": 1,
                        "hidden_layer_shapes": [2, 1],
                        "layer_activation_funcs": ["tanh", "sigmoid"],
                        "loss_func": "mse",
                    },
                    "output_modifier_name": None,
                    "training_config": None,
                },
                weights_0=np.ones((3, 1), dtype=float),
                biases_0=np.zeros(2, dtype=float),
                weights_1=np.ones((1, 2), dtype=float),
                biases_1=np.zeros(1, dtype=float),
            )

            with self.assertRaisesRegex(ValueError, "weights_0"):
                load_network(path)

    def _write_archive(
        self,
        path: Path,
        *,
        metadata: dict | None = None,
        metadata_bytes: bytes | None = None,
        **arrays: np.ndarray,
    ) -> None:
        if metadata_bytes is None:
            metadata_bytes = json.dumps(
                metadata,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")

        archive_arrays = {
            "__ffnnpy_metadata__": np.frombuffer(metadata_bytes, dtype=np.uint8).copy(),
            **arrays,
        }
        with path.open("wb") as handle:
            np.savez_compressed(handle, **archive_arrays)


if __name__ == "__main__":
    unittest.main()
