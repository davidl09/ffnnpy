from __future__ import annotations

import importlib.util
import unittest

import numpy as np

from neural_net import (
    AcceleratedRuntime,
    AcceleratedTrainingConfig,
    ActivationFunc,
    TrainingConfig,
    build_accelerated_network,
    build_random_network,
    fit_dataset,
    fit_dataset_accelerated,
    fit_function_accelerated,
    predict_dataset_accelerated,
)
from neural_net.backend import get_loss_func


HAS_NUMBA = importlib.util.find_spec("numba") is not None


class AcceleratedBackendTests(unittest.TestCase):
    def test_forward_batch_shape(self):
        network = build_accelerated_network(
            input_layer_dim=1,
            hidden_layer_shapes=(8, 8, 1),
            activation=ActivationFunc.tanh,
            seed=0,
        )

        outputs = network.forward_batch(np.linspace(-1.0, 1.0, 11, dtype=float).reshape(-1, 1))

        self.assertEqual(outputs.shape, (11, 1))

    def test_predict_dataset_accelerated_shape(self):
        network = build_accelerated_network(
            input_layer_dim=1,
            hidden_layer_shapes=(8, 8, 1),
            activation=ActivationFunc.tanh,
            seed=0,
        )
        xs = np.linspace(-np.pi, np.pi, 33, dtype=float).reshape(-1, 1)

        outputs = predict_dataset_accelerated(network, xs)

        self.assertEqual(outputs.shape, (33, 1))

    def test_fit_dataset_accelerated_reduces_loss(self):
        network = build_accelerated_network(
            input_layer_dim=1,
            hidden_layer_shapes=(16, 16, 1),
            activation=ActivationFunc.tanh,
            seed=0,
        )
        xs = np.linspace(-np.pi, np.pi, 512, dtype=float).reshape(-1, 1)
        ys = np.sin(xs)
        loss_fn = get_loss_func(network.config.loss_func)
        initial_loss = float(loss_fn(ys, predict_dataset_accelerated(network, xs)))

        result = fit_dataset_accelerated(
            network,
            xs,
            ys,
            config=AcceleratedTrainingConfig(
                learning_rate=0.02,
                max_power=9,
                evaluation_points=128,
                seed=0,
                batch_size=64,
                runtime=AcceleratedRuntime.numpy,
            ),
        )

        final_loss = result.losses[result.milestone_steps[-1]]
        self.assertLess(final_loss, initial_loss)

    def test_fit_function_accelerated_reduces_loss(self):
        network = build_accelerated_network(
            input_layer_dim=1,
            hidden_layer_shapes=(16, 16, 1),
            activation=ActivationFunc.tanh,
            seed=0,
        )
        evaluation_inputs = np.linspace(-np.pi, np.pi, 256, dtype=float).reshape(-1, 1)
        evaluation_targets = np.sin(evaluation_inputs)
        loss_fn = get_loss_func(network.config.loss_func)
        initial_loss = float(loss_fn(evaluation_targets, predict_dataset_accelerated(network, evaluation_inputs)))

        result = fit_function_accelerated(
            network,
            np.sin,
            config=AcceleratedTrainingConfig(
                learning_rate=0.02,
                max_power=9,
                evaluation_points=256,
                seed=0,
                batch_size=64,
                runtime=AcceleratedRuntime.numpy,
            ),
        )

        final_loss = result.losses[result.milestone_steps[-1]]
        self.assertLess(final_loss, initial_loss)

    def test_fit_function_accelerated_rejects_non_vectorized_target(self):
        network = build_accelerated_network(
            input_layer_dim=1,
            hidden_layer_shapes=(8, 8, 1),
            activation=ActivationFunc.tanh,
            seed=0,
        )

        def bad_target(_):
            return 1.0

        with self.assertRaisesRegex(ValueError, "requires target_func"):
            fit_function_accelerated(
                network,
                bad_target,
                config=AcceleratedTrainingConfig(max_power=1, batch_size=8, runtime=AcceleratedRuntime.numpy),
            )

    def test_fit_function_accelerated_rejects_wrong_sample_count(self):
        network = build_accelerated_network(
            input_layer_dim=1,
            hidden_layer_shapes=(8, 8, 1),
            activation=ActivationFunc.tanh,
            seed=0,
        )

        def bad_target(_):
            return np.array([0.0, 1.0], dtype=float)

        with self.assertRaisesRegex(ValueError, "one scalar output per input sample"):
            fit_function_accelerated(
                network,
                bad_target,
                config=AcceleratedTrainingConfig(max_power=1, batch_size=8, runtime=AcceleratedRuntime.numpy),
            )

    def test_reference_and_accelerated_match_for_batch_size_one(self):
        xs = np.linspace(-1.0, 1.0, 64, dtype=float).reshape(-1, 1)
        ys = np.sin(xs)

        reference_network = build_random_network(
            input_layer_dim=1,
            hidden_layer_shapes=(8, 8, 1),
            activation=ActivationFunc.tanh,
            seed=0,
        )
        accelerated_network = build_accelerated_network(
            input_layer_dim=1,
            hidden_layer_shapes=(8, 8, 1),
            activation=ActivationFunc.tanh,
            seed=0,
            runtime=AcceleratedRuntime.numpy,
        )

        reference_result = fit_dataset(
            reference_network,
            xs,
            ys,
            config=TrainingConfig(learning_rate=0.02, max_power=7, evaluation_points=64, seed=0),
            evaluation_inputs=xs,
            evaluation_targets=ys,
        )
        accelerated_result = fit_dataset_accelerated(
            accelerated_network,
            xs,
            ys,
            config=AcceleratedTrainingConfig(
                learning_rate=0.02,
                max_power=7,
                evaluation_points=64,
                seed=0,
                batch_size=1,
                runtime=AcceleratedRuntime.numpy,
            ),
            evaluation_inputs=xs,
            evaluation_targets=ys,
        )

        self.assertEqual(reference_result.milestone_steps, accelerated_result.milestone_steps)
        self.assertEqual(sorted(reference_result.snapshots), sorted(accelerated_result.snapshots))
        self.assertEqual(
            reference_result.snapshots[reference_result.milestone_steps[-1]].shape,
            accelerated_result.snapshots[accelerated_result.milestone_steps[-1]].shape,
        )
        self.assertTrue(
            np.allclose(
                reference_result.snapshots[reference_result.milestone_steps[-1]],
                accelerated_result.snapshots[accelerated_result.milestone_steps[-1]],
                atol=1e-8,
                rtol=1e-8,
            )
        )
        self.assertAlmostEqual(
            reference_result.losses[reference_result.milestone_steps[-1]],
            accelerated_result.losses[accelerated_result.milestone_steps[-1]],
            places=10,
        )

    def test_numba_runtime_behavior_matches_environment(self):
        network = build_accelerated_network(
            input_layer_dim=1,
            hidden_layer_shapes=(8, 8, 1),
            activation=ActivationFunc.tanh,
            seed=0,
        )
        xs = np.linspace(-1.0, 1.0, 8, dtype=float).reshape(-1, 1)

        if HAS_NUMBA:
            outputs = predict_dataset_accelerated(network, xs, runtime=AcceleratedRuntime.numba)
            self.assertEqual(outputs.shape, (8, 1))
        else:
            with self.assertRaisesRegex(RuntimeError, "numba is not installed"):
                predict_dataset_accelerated(network, xs, runtime=AcceleratedRuntime.numba)


if __name__ == "__main__":
    unittest.main()
