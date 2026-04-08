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
    predict_dataset,
    predict_dataset_accelerated,
    powers_of_two_milestones,
)
from neural_net.backend import get_loss_func


HAS_NUMBA = importlib.util.find_spec("numba") is not None


def _boolean_threshold(output: np.ndarray) -> bool:
    return bool(output[0] >= 0.5)


class OutputModifierTests(unittest.TestCase):
    def test_predict_dataset_reference_shape_unchanged_without_modifier(self):
        network = build_random_network(
            input_layer_dim=1,
            hidden_layer_shapes=(8, 8, 1),
            activation=ActivationFunc.tanh,
            seed=0,
        )
        xs = np.linspace(-1.0, 1.0, 17, dtype=float).reshape(-1, 1)

        outputs = predict_dataset(network, xs)

        self.assertEqual(outputs.shape, (17, 1))
        self.assertTrue(np.issubdtype(outputs.dtype, np.floating))

    def test_reference_inference_applies_output_modifier(self):
        network = build_random_network(
            input_layer_dim=1,
            hidden_layer_shapes=(1,),
            activation=ActivationFunc.sigmoid,
            seed=0,
            output_modifier=_boolean_threshold,
        )
        network.weights[0][:] = 0.0
        network.biases[0][:] = 0.6
        xs = np.array([[-1.0], [0.0], [1.0]], dtype=float)

        output = network.fast_forward_pass(np.array([0.0], dtype=float))
        outputs = predict_dataset(network, xs)

        self.assertIs(output, True)
        self.assertEqual(outputs.shape, (3,))
        self.assertEqual(outputs.dtype, np.bool_)
        self.assertTrue(np.all(outputs))

    def test_predict_dataset_rejects_inconsistent_output_modifier_shapes(self):
        def inconsistent_shape(output: np.ndarray) -> np.ndarray:
            if output[0] < 0.5:
                return np.array([output[0]], dtype=float)
            return np.array([output[0], output[0]], dtype=float)

        network = build_random_network(
            input_layer_dim=1,
            hidden_layer_shapes=(1,),
            activation=ActivationFunc.sigmoid,
            seed=0,
            output_modifier=inconsistent_shape,
        )
        network.weights[0][:] = 10.0
        network.biases[0][:] = -5.0

        with self.assertRaisesRegex(ValueError, "consistent shape"):
            predict_dataset(network, np.array([[0.0], [1.0]], dtype=float))

    def test_fit_dataset_uses_raw_outputs_when_modifier_is_configured(self):
        network = build_random_network(
            input_layer_dim=1,
            hidden_layer_shapes=(16, 16, 1),
            activation=ActivationFunc.tanh,
            seed=0,
            output_modifier=lambda output: bool(output[0] >= 0.0),
        )
        xs = np.linspace(-np.pi, np.pi, 256, dtype=float).reshape(-1, 1)
        ys = np.sin(xs)
        loss_fn = get_loss_func(network.config.loss_func)
        initial_loss = float(
            loss_fn(
                ys,
                np.array([network._raw_forward_pass(sample) for sample in xs], dtype=float),
            )
        )

        result = fit_dataset(
            network,
            xs,
            ys,
            config=TrainingConfig(
                learning_rate=0.02,
                milestones=powers_of_two_milestones(9),
                evaluation_points=128,
                seed=0,
            ),
            evaluation_inputs=xs,
            evaluation_targets=ys,
        )

        final_loss = result.losses[result.milestones[-1]]
        self.assertLess(final_loss, initial_loss)
        self.assertEqual(result.snapshots[result.milestones[-1]].shape, (256, 1))
        self.assertTrue(np.issubdtype(result.snapshots[result.milestones[-1]].dtype, np.floating))

    def test_accelerated_inference_applies_output_modifier_numpy(self):
        network = build_accelerated_network(
            input_layer_dim=1,
            hidden_layer_shapes=(1,),
            activation=ActivationFunc.sigmoid,
            seed=0,
            runtime=AcceleratedRuntime.numpy,
            output_modifier=_boolean_threshold,
        )
        network.weights[0][:] = 0.0
        network.biases[0][:] = 0.6
        xs = np.array([[-1.0], [0.0], [1.0]], dtype=float)

        output = network.fast_forward_pass(np.array([0.0], dtype=float))
        outputs = predict_dataset_accelerated(network, xs, runtime=AcceleratedRuntime.numpy)

        self.assertIs(output, True)
        self.assertEqual(outputs.shape, (3,))
        self.assertEqual(outputs.dtype, np.bool_)
        self.assertTrue(np.all(outputs))

    @unittest.skipUnless(HAS_NUMBA, "numba is not installed")
    def test_accelerated_inference_applies_output_modifier_numba(self):
        network = build_accelerated_network(
            input_layer_dim=1,
            hidden_layer_shapes=(1,),
            activation=ActivationFunc.sigmoid,
            seed=0,
            runtime=AcceleratedRuntime.numba,
            output_modifier=_boolean_threshold,
        )
        network.weights[0][:] = 0.0
        network.biases[0][:] = 0.6
        xs = np.array([[-1.0], [0.0], [1.0]], dtype=float)

        output = network.fast_forward_pass(np.array([0.0], dtype=float))
        outputs = predict_dataset_accelerated(network, xs, runtime=AcceleratedRuntime.numba)

        self.assertIs(output, True)
        self.assertEqual(outputs.shape, (3,))
        self.assertEqual(outputs.dtype, np.bool_)
        self.assertTrue(np.all(outputs))

    def test_fit_dataset_accelerated_uses_raw_outputs_when_modifier_is_configured(self):
        network = build_accelerated_network(
            input_layer_dim=1,
            hidden_layer_shapes=(16, 16, 1),
            activation=ActivationFunc.tanh,
            seed=0,
            runtime=AcceleratedRuntime.numpy,
            output_modifier=lambda output: bool(output[0] >= 0.0),
        )
        xs = np.linspace(-np.pi, np.pi, 512, dtype=float).reshape(-1, 1)
        ys = np.sin(xs)
        loss_fn = get_loss_func(network.config.loss_func)
        initial_loss = float(
            loss_fn(
                ys,
                network._forward_batch_raw(xs, runtime=AcceleratedRuntime.numpy),
            )
        )

        result = fit_dataset_accelerated(
            network,
            xs,
            ys,
            config=AcceleratedTrainingConfig(
                learning_rate=0.02,
                milestones=powers_of_two_milestones(9),
                evaluation_points=128,
                seed=0,
                batch_size=64,
                runtime=AcceleratedRuntime.numpy,
            ),
        )

        final_loss = result.losses[result.milestones[-1]]
        self.assertLess(final_loss, initial_loss)
        self.assertEqual(result.snapshots[result.milestones[-1]].shape, (512, 1))
        self.assertTrue(np.issubdtype(result.snapshots[result.milestones[-1]].dtype, np.floating))


if __name__ == "__main__":
    unittest.main()
