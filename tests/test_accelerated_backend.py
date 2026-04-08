from __future__ import annotations

import importlib.util
import math
import unittest

import numpy as np

from neural_net import (
    AcceleratedRuntime,
    AcceleratedTrainingConfig,
    ActivationFunc,
    FFNNConfig,
    LossFunc,
    TrainingConfig,
    build_accelerated_network,
    build_random_network,
    fit_dataset,
    fit_dataset_accelerated,
    fit_function,
    fit_function_accelerated,
    predict_dataset,
    predict_dataset_accelerated,
)
from neural_net.backend import get_loss_func


HAS_NUMBA = importlib.util.find_spec("numba") is not None


class AcceleratedBackendTests(unittest.TestCase):
    def test_config_expands_single_activation_to_all_layers(self):
        config = FFNNConfig(
            input_layer_dim=1,
            hidden_layer_count=3,
            hidden_layer_shapes=(4, 4, 1),
            activation_func=ActivationFunc.tanh,
        )

        self.assertEqual(
            config.layer_activation_funcs,
            (ActivationFunc.tanh, ActivationFunc.tanh, ActivationFunc.tanh),
        )

    def test_config_rejects_wrong_per_layer_activation_count(self):
        with self.assertRaisesRegex(ValueError, "activation_func sequence length must match hidden_layer_count"):
            FFNNConfig(
                input_layer_dim=1,
                hidden_layer_count=3,
                hidden_layer_shapes=(4, 4, 1),
                activation_func=(ActivationFunc.relu, ActivationFunc.tanh),
            )

    def test_config_rejects_cross_entropy_without_sigmoid_output(self):
        with self.assertRaisesRegex(ValueError, "sigmoid output activation"):
            FFNNConfig(
                input_layer_dim=1,
                hidden_layer_count=2,
                hidden_layer_shapes=(4, 1),
                activation_func=(ActivationFunc.tanh, ActivationFunc.tanh),
                loss_func=LossFunc.cross_entropy,
            )

    def test_builders_expose_public_loss_selection(self):
        reference_network = build_random_network(
            input_layer_dim=1,
            hidden_layer_shapes=(4, 1),
            activation=(ActivationFunc.tanh, ActivationFunc.sigmoid),
            loss_func="cross_entropy",
            seed=0,
        )
        accelerated_network = build_accelerated_network(
            input_layer_dim=1,
            hidden_layer_shapes=(4, 1),
            activation=(ActivationFunc.tanh, ActivationFunc.sigmoid),
            loss_func=LossFunc.cross_entropy,
            seed=0,
            runtime=AcceleratedRuntime.numpy,
        )

        self.assertIs(reference_network.config.loss_func, LossFunc.cross_entropy)
        self.assertIs(accelerated_network.config.loss_func, LossFunc.cross_entropy)

    def test_cross_entropy_loss_matches_manual_formula(self):
        targets = np.array([[1.0], [0.0], [1.0]], dtype=float)
        predictions = np.array([[0.9], [0.2], [0.8]], dtype=float)
        expected = -np.mean(
            targets.reshape(-1) * np.log(predictions.reshape(-1))
            + (1.0 - targets.reshape(-1)) * np.log(1.0 - predictions.reshape(-1))
        )

        loss_fn = get_loss_func(LossFunc.cross_entropy)

        self.assertAlmostEqual(float(loss_fn(targets, predictions)), float(expected), places=12)

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

    def test_fit_function_reference_accepts_scalar_target(self):
        network = build_random_network(
            input_layer_dim=1,
            hidden_layer_shapes=(16, 16, 1),
            activation=ActivationFunc.tanh,
            seed=0,
        )
        evaluation_inputs = np.linspace(-np.pi, np.pi, 256, dtype=float).reshape(-1, 1)
        evaluation_targets = np.sin(evaluation_inputs)
        loss_fn = get_loss_func(network.config.loss_func)
        initial_loss = float(loss_fn(evaluation_targets, predict_dataset(network, evaluation_inputs)))

        result = fit_function(
            network,
            math.sin,
            config=TrainingConfig(
                learning_rate=0.02,
                max_power=9,
                evaluation_points=256,
                seed=0,
            ),
        )

        final_loss = result.losses[result.milestone_steps[-1]]
        self.assertLess(final_loss, initial_loss)

    def test_fit_function_reference_accepts_vectorized_target(self):
        network = build_random_network(
            input_layer_dim=1,
            hidden_layer_shapes=(16, 16, 1),
            activation=ActivationFunc.tanh,
            seed=0,
        )

        result = fit_function(
            network,
            np.sin,
            config=TrainingConfig(
                learning_rate=0.02,
                max_power=7,
                evaluation_points=64,
                seed=0,
            ),
        )

        self.assertEqual(
            result.snapshots[result.milestone_steps[-1]].shape,
            (64, 1),
        )

    def test_fit_function_reference_rejects_invalid_target(self):
        network = build_random_network(
            input_layer_dim=1,
            hidden_layer_shapes=(8, 8, 1),
            activation=ActivationFunc.tanh,
            seed=0,
        )

        def bad_target(_):
            return np.array([0.0, 1.0], dtype=float)

        with self.assertRaisesRegex(ValueError, "accept either a float and return a scalar"):
            fit_function(
                network,
                bad_target,
                config=TrainingConfig(max_power=1, evaluation_points=8, seed=0),
            )

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

    def test_fit_dataset_accelerated_inherits_network_runtime_when_config_runtime_is_omitted(self):
        network = build_accelerated_network(
            input_layer_dim=1,
            hidden_layer_shapes=(8, 8, 1),
            activation=ActivationFunc.tanh,
            seed=0,
            runtime=AcceleratedRuntime.numpy,
        )
        xs = np.linspace(-1.0, 1.0, 16, dtype=float).reshape(-1, 1)
        ys = np.sin(xs)

        fit_dataset_accelerated(
            network,
            xs,
            ys,
            config=AcceleratedTrainingConfig(
                learning_rate=0.02,
                max_power=0,
                evaluation_points=16,
                seed=0,
                batch_size=4,
            ),
        )

        self.assertIsNone(network._numba_weights)
        self.assertIsNone(network._numba_biases)

    def test_fit_function_accelerated_explicit_runtime_overrides_network_runtime(self):
        network = build_accelerated_network(
            input_layer_dim=1,
            hidden_layer_shapes=(8, 8, 1),
            activation=ActivationFunc.tanh,
            seed=0,
            runtime=AcceleratedRuntime.numba,
        )

        fit_function_accelerated(
            network,
            np.sin,
            config=AcceleratedTrainingConfig(
                learning_rate=0.02,
                max_power=0,
                evaluation_points=16,
                seed=0,
                batch_size=4,
                runtime=AcceleratedRuntime.numpy,
            ),
        )

        self.assertIsNone(network._numba_weights)
        self.assertIsNone(network._numba_biases)

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

    def test_accelerated_fast_forward_pass_accepts_single_row_input(self):
        network = build_accelerated_network(
            input_layer_dim=1,
            hidden_layer_shapes=(8, 8, 1),
            activation=ActivationFunc.tanh,
            seed=0,
            runtime=AcceleratedRuntime.numpy,
        )
        sample = np.array([0.25], dtype=float)
        sample_row = sample.reshape(1, -1)

        output_vector = np.asarray(network.fast_forward_pass(sample), dtype=float)
        output_row = np.asarray(network.fast_forward_pass(sample_row), dtype=float)

        self.assertTrue(np.allclose(output_vector, output_row, atol=1e-10, rtol=1e-10))

    def test_accelerated_fast_forward_pass_rejects_batched_input(self):
        network = build_accelerated_network(
            input_layer_dim=1,
            hidden_layer_shapes=(8, 8, 1),
            activation=ActivationFunc.tanh,
            seed=0,
            runtime=AcceleratedRuntime.numpy,
        )

        with self.assertRaisesRegex(ValueError, "expects exactly one sample"):
            network.fast_forward_pass(
                np.array([[0.0], [1.0]], dtype=float)
            )

    def test_reference_and_accelerated_match_with_per_layer_activations(self):
        activations = (
            ActivationFunc.relu,
            ActivationFunc.tanh,
            ActivationFunc.sigmoid,
        )
        xs = np.linspace(-1.0, 1.0, 16, dtype=float).reshape(-1, 1)
        ys = np.sin(xs)

        reference_network = build_random_network(
            input_layer_dim=1,
            hidden_layer_shapes=(8, 8, 1),
            activation=activations,
            seed=0,
        )
        accelerated_network = build_accelerated_network(
            input_layer_dim=1,
            hidden_layer_shapes=(8, 8, 1),
            activation=activations,
            seed=0,
            runtime=AcceleratedRuntime.numpy,
        )

        self.assertTrue(
            np.allclose(
                predict_dataset(reference_network, xs),
                predict_dataset_accelerated(accelerated_network, xs),
                atol=1e-10,
                rtol=1e-10,
            )
        )

        reference_network.fast_backward_pass(xs[0], ys[0], 0.02)
        accelerated_network.train_batch(xs[:1], ys[:1], 0.02, runtime=AcceleratedRuntime.numpy)

        for reference_weights, accelerated_weights in zip(reference_network.weights, accelerated_network.weights):
            self.assertTrue(np.allclose(reference_weights, accelerated_weights, atol=1e-10, rtol=1e-10))
        for reference_biases, accelerated_biases in zip(reference_network.biases, accelerated_network.biases):
            self.assertTrue(np.allclose(reference_biases, accelerated_biases, atol=1e-10, rtol=1e-10))

    def test_reference_and_accelerated_match_with_cross_entropy(self):
        activations = (
            ActivationFunc.relu,
            ActivationFunc.tanh,
            ActivationFunc.sigmoid,
        )
        xs = np.linspace(-1.0, 1.0, 16, dtype=float).reshape(-1, 1)
        ys = (xs >= 0.0).astype(float)

        reference_network = build_random_network(
            input_layer_dim=1,
            hidden_layer_shapes=(8, 8, 1),
            activation=activations,
            loss_func=LossFunc.cross_entropy,
            seed=0,
        )
        accelerated_network = build_accelerated_network(
            input_layer_dim=1,
            hidden_layer_shapes=(8, 8, 1),
            activation=activations,
            loss_func=LossFunc.cross_entropy,
            seed=0,
            runtime=AcceleratedRuntime.numpy,
        )

        self.assertTrue(
            np.allclose(
                predict_dataset(reference_network, xs),
                predict_dataset_accelerated(accelerated_network, xs),
                atol=1e-10,
                rtol=1e-10,
            )
        )

        reference_network.fast_backward_pass(xs[0], ys[0], 0.02)
        accelerated_network.train_batch(xs[:1], ys[:1], 0.02, runtime=AcceleratedRuntime.numpy)

        for reference_weights, accelerated_weights in zip(reference_network.weights, accelerated_network.weights):
            self.assertTrue(np.allclose(reference_weights, accelerated_weights, atol=1e-10, rtol=1e-10))
        for reference_biases, accelerated_biases in zip(reference_network.biases, accelerated_network.biases):
            self.assertTrue(np.allclose(reference_biases, accelerated_biases, atol=1e-10, rtol=1e-10))

    @unittest.skipUnless(HAS_NUMBA, "numba is not installed")
    def test_numba_mixed_activations_match_numpy(self):
        activations = (
            ActivationFunc.relu,
            ActivationFunc.tanh,
            ActivationFunc.sigmoid,
        )
        xs = np.linspace(-1.0, 1.0, 8, dtype=float).reshape(-1, 1)
        ys = np.sin(xs)
        numpy_network = build_accelerated_network(
            input_layer_dim=1,
            hidden_layer_shapes=(8, 8, 1),
            activation=activations,
            seed=0,
            runtime=AcceleratedRuntime.numpy,
        )
        numba_network = build_accelerated_network(
            input_layer_dim=1,
            hidden_layer_shapes=(8, 8, 1),
            activation=activations,
            seed=0,
            runtime=AcceleratedRuntime.numba,
        )

        self.assertTrue(
            np.allclose(
                predict_dataset_accelerated(numpy_network, xs, runtime=AcceleratedRuntime.numpy),
                predict_dataset_accelerated(numba_network, xs, runtime=AcceleratedRuntime.numba),
                atol=1e-10,
                rtol=1e-10,
            )
        )

        numpy_predictions = numpy_network.train_batch(xs[:2], ys[:2], 0.02, runtime=AcceleratedRuntime.numpy)
        numba_predictions = numba_network.train_batch(xs[:2], ys[:2], 0.02, runtime=AcceleratedRuntime.numba)

        self.assertTrue(np.allclose(numpy_predictions, numba_predictions, atol=1e-10, rtol=1e-10))
        for numpy_weights, numba_weights in zip(numpy_network.weights, numba_network.weights):
            self.assertTrue(np.allclose(numpy_weights, numba_weights, atol=1e-10, rtol=1e-10))
        for numpy_biases, numba_biases in zip(numpy_network.biases, numba_network.biases):
            self.assertTrue(np.allclose(numpy_biases, numba_biases, atol=1e-10, rtol=1e-10))

    @unittest.skipUnless(HAS_NUMBA, "numba is not installed")
    def test_numba_cross_entropy_matches_numpy(self):
        activations = (
            ActivationFunc.relu,
            ActivationFunc.tanh,
            ActivationFunc.sigmoid,
        )
        xs = np.array([[-1.0], [1.0]], dtype=float)
        ys = np.array([[0.0], [1.0]], dtype=float)
        numpy_network = build_accelerated_network(
            input_layer_dim=1,
            hidden_layer_shapes=(8, 8, 1),
            activation=activations,
            loss_func=LossFunc.cross_entropy,
            seed=0,
            runtime=AcceleratedRuntime.numpy,
        )
        numba_network = build_accelerated_network(
            input_layer_dim=1,
            hidden_layer_shapes=(8, 8, 1),
            activation=activations,
            loss_func=LossFunc.cross_entropy,
            seed=0,
            runtime=AcceleratedRuntime.numba,
        )

        numpy_predictions = numpy_network.train_batch(
            xs,
            ys,
            0.02,
            runtime=AcceleratedRuntime.numpy,
        )
        numba_predictions = numba_network.train_batch(
            xs,
            ys,
            0.02,
            runtime=AcceleratedRuntime.numba,
        )

        self.assertTrue(np.allclose(numpy_predictions, numba_predictions, atol=1e-10, rtol=1e-10))
        for numpy_weights, numba_weights in zip(numpy_network.weights, numba_network.weights):
            self.assertTrue(np.allclose(numpy_weights, numba_weights, atol=1e-10, rtol=1e-10))
        for numpy_biases, numba_biases in zip(numpy_network.biases, numba_network.biases):
            self.assertTrue(np.allclose(numpy_biases, numba_biases, atol=1e-10, rtol=1e-10))

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

    def test_fit_dataset_rejects_empty_training_data(self):
        network = build_random_network(
            input_layer_dim=1,
            hidden_layer_shapes=(8, 8, 1),
            activation=ActivationFunc.tanh,
            seed=0,
        )

        with self.assertRaisesRegex(ValueError, "contain at least one sample"):
            fit_dataset(
                network,
                np.empty((0, 1), dtype=float),
                np.empty((0, 1), dtype=float),
                config=TrainingConfig(max_power=1, evaluation_points=8, seed=0),
            )

    def test_fit_dataset_accelerated_rejects_empty_training_data(self):
        network = build_accelerated_network(
            input_layer_dim=1,
            hidden_layer_shapes=(8, 8, 1),
            activation=ActivationFunc.tanh,
            seed=0,
            runtime=AcceleratedRuntime.numpy,
        )

        with self.assertRaisesRegex(ValueError, "contain at least one sample"):
            fit_dataset_accelerated(
                network,
                np.empty((0, 1), dtype=float),
                np.empty((0, 1), dtype=float),
                config=AcceleratedTrainingConfig(
                    max_power=1,
                    evaluation_points=8,
                    seed=0,
                    batch_size=4,
                    runtime=AcceleratedRuntime.numpy,
                ),
            )

    def test_fit_dataset_rejects_empty_evaluation_data(self):
        network = build_random_network(
            input_layer_dim=1,
            hidden_layer_shapes=(8, 8, 1),
            activation=ActivationFunc.tanh,
            seed=0,
        )
        xs = np.linspace(-1.0, 1.0, 8, dtype=float).reshape(-1, 1)
        ys = np.sin(xs)

        with self.assertRaisesRegex(ValueError, "contain at least one sample"):
            fit_dataset(
                network,
                xs,
                ys,
                config=TrainingConfig(max_power=1, evaluation_points=8, seed=0),
                evaluation_inputs=np.empty((0, 1), dtype=float),
                evaluation_targets=np.empty((0, 1), dtype=float),
            )

    def test_fit_dataset_accelerated_rejects_empty_evaluation_data(self):
        network = build_accelerated_network(
            input_layer_dim=1,
            hidden_layer_shapes=(8, 8, 1),
            activation=ActivationFunc.tanh,
            seed=0,
            runtime=AcceleratedRuntime.numpy,
        )
        xs = np.linspace(-1.0, 1.0, 8, dtype=float).reshape(-1, 1)
        ys = np.sin(xs)

        with self.assertRaisesRegex(ValueError, "contain at least one sample"):
            fit_dataset_accelerated(
                network,
                xs,
                ys,
                config=AcceleratedTrainingConfig(
                    max_power=1,
                    evaluation_points=8,
                    seed=0,
                    batch_size=4,
                    runtime=AcceleratedRuntime.numpy,
                ),
                evaluation_inputs=np.empty((0, 1), dtype=float),
                evaluation_targets=np.empty((0, 1), dtype=float),
            )


if __name__ == "__main__":
    unittest.main()
