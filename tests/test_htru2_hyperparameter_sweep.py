from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

from ffnnpy.neural_net import AcceleratedRuntime


ROOT = Path(__file__).resolve().parents[2]
SWEEP_SCRIPT_PATH = ROOT / "experiments" / "htru2_hyperparameter_sweep.py"


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class Htru2HyperparameterSweepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module("htru2_hyperparameter_sweep", SWEEP_SCRIPT_PATH)

    def test_build_training_config_matches_train_model_schema(self):
        spec = self.module.RunSpec(
            stage="final_confirm",
            architecture_name="medium_tapered",
            architecture_shape=(32, 16, 1),
            train_fraction=0.8,
            learning_rate=0.01,
            init_seed=47,
            split_seed=20260407,
            max_power=16,
            batch_size=128,
        )

        config = self.module.build_training_config(spec)

        self.assertEqual(config.learning_rate, 0.01)
        self.assertEqual(config.max_power, 16)
        self.assertEqual(config.evaluation_points, self.module.DEFAULT_EVALUATION_POINT_COUNT)
        self.assertEqual(config.seed, 47)
        self.assertEqual(config.batch_size, 128)
        self.assertEqual(config.runtime, AcceleratedRuntime.numba)

    def test_build_model_hyperparameters_exports_expected_json_payload(self):
        spec = self.module.RunSpec(
            stage="final_confirm",
            architecture_name="medium_tapered",
            architecture_shape=(32, 16, 1),
            train_fraction=0.8,
            learning_rate=0.01,
            init_seed=47,
            split_seed=20260407,
            max_power=16,
            batch_size=128,
        )

        hyperparameters = self.module.build_model_hyperparameters(spec)

        self.assertEqual(
            hyperparameters.to_json_dict(),
            {
                "train_fraction": 0.8,
                "split_seed": 20260407,
                "hidden_layer_shapes": [32, 16, 1],
                "activation": ["sigmoid"],
                "seed": 47,
                "learning_rate": 0.01,
                "max_power": 16,
                "evaluation_points": self.module.DEFAULT_EVALUATION_POINT_COUNT,
                "batch_size": 128,
                "runtime": "numba",
            },
        )


if __name__ == "__main__":
    unittest.main()
