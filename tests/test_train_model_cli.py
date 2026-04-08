from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ffnnpy.neural_net import (
    AcceleratedRuntime,
    AcceleratedTrainingConfig,
    ActivationFunc,
    load_network,
)


ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT_PATH = ROOT / "train_model.py"
CONFIGURE_SCRIPT_PATH = ROOT / "configure_model.py"
STATS_SCRIPT_PATH = ROOT / "experiments" / "htru2_saved_model_stats.py"
DATASET_PATH = ROOT / "htru2" / "HTRU_2.arff"


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_configure_model(temp_dir: str, run_name: str, *extra_args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(CONFIGURE_SCRIPT_PATH),
            run_name,
            *extra_args,
        ],
        cwd=temp_dir,
        capture_output=True,
        text=True,
    )


class ConfigureModelCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module("configure_model", CONFIGURE_SCRIPT_PATH)

    def test_cli_writes_hyperparams_json_with_requested_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_name = "configured-run"
            hyperparams_path = Path(temp_dir) / "results" / run_name / "hyperparams.json"

            process = run_configure_model(
                temp_dir,
                run_name,
                "--runtime",
                "numpy",
                "--max-power",
                "1",
                "--batch-size",
                "32",
                "--hidden-layer-shapes",
                "4",
                "1",
                "--activation",
                "tanh",
                "sigmoid",
                "--seed",
                "5",
                "--split-seed",
                "7",
                "--train-fraction",
                "0.7",
                "--learning-rate",
                "0.02",
                "--evaluation-points",
                "64",
            )

            self.assertEqual(process.returncode, 0, msg=process.stderr)
            self.assertTrue(hyperparams_path.exists())
            self.assertIn(str(hyperparams_path.resolve()), process.stdout)
            self.assertEqual(
                json.loads(hyperparams_path.read_text(encoding="utf-8")),
                {
                    "train_fraction": 0.7,
                    "split_seed": 7,
                    "hidden_layer_shapes": [4, 1],
                    "activation": ["tanh", "sigmoid"],
                    "seed": 5,
                    "learning_rate": 0.02,
                    "max_power": 1,
                    "evaluation_points": 64,
                    "batch_size": 32,
                    "runtime": "numpy",
                },
            )


class TrainModelCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module("train_model", TRAIN_SCRIPT_PATH)
        cls.stats_module = load_module("htru2_saved_model_stats", STATS_SCRIPT_PATH)

    def test_default_output_path_uses_named_results_directory(self):
        path = self.module.resolve_output_path(None, "demo-run")

        self.assertEqual(
            path,
            Path("results") / "demo-run" / "model.ffnnpy",
        )

    def test_default_hyperparams_path_uses_named_results_directory(self):
        self.assertEqual(
            self.module.resolve_hyperparams_path("demo-run"),
            Path("results") / "demo-run" / "hyperparams.json",
        )

    def test_activation_resolution_supports_broadcast_and_per_layer_values(self):
        self.assertEqual(
            self.module.resolve_activation_sequence(["sigmoid"], 2),
            ActivationFunc.sigmoid,
        )
        self.assertEqual(
            self.module.resolve_activation_sequence(["relu", "sigmoid"], 2),
            (ActivationFunc.relu, ActivationFunc.sigmoid),
        )

    def test_activation_resolution_rejects_invalid_activation_count(self):
        with self.assertRaisesRegex(ValueError, "activation count"):
            self.module.resolve_activation_sequence(["relu", "tanh"], 3)

    def test_stratified_split_keeps_both_classes_in_train_and_test(self):
        features = np.arange(12, dtype=float).reshape(6, 2)
        labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)

        train_x, train_y, test_x, test_y = self.module.stratified_split(
            features,
            labels,
            train_fraction=0.5,
            split_seed=3,
        )

        self.assertEqual(train_x.shape, (4, 2))
        self.assertEqual(test_x.shape, (2, 2))
        self.assertEqual(set(train_y.tolist()), {0, 1})
        self.assertEqual(set(test_y.tolist()), {0, 1})

    def test_cli_smoke_run_reads_config_and_saves_model_with_requested_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_name = "smoke-run"
            output_path = Path(temp_dir) / "smoke.ffnnpy"
            configure_process = run_configure_model(
                temp_dir,
                run_name,
                "--runtime",
                "numpy",
                "--max-power",
                "1",
                "--batch-size",
                "32",
                "--hidden-layer-shapes",
                "4",
                "1",
                "--activation",
                "tanh",
                "sigmoid",
                "--seed",
                "5",
                "--split-seed",
                "7",
                "--train-fraction",
                "0.7",
                "--learning-rate",
                "0.02",
                "--evaluation-points",
                "64",
            )
            self.assertEqual(configure_process.returncode, 0, msg=configure_process.stderr)

            process = subprocess.run(
                [
                    sys.executable,
                    str(TRAIN_SCRIPT_PATH),
                    run_name,
                    "--dataset-path",
                    str(DATASET_PATH),
                    "--output-path",
                    str(output_path),
                ],
                cwd=temp_dir,
                capture_output=True,
                text=True,
            )

            self.assertEqual(process.returncode, 0, msg=process.stderr)
            self.assertIn("Loaded hyperparameters:", process.stdout)
            self.assertIn("Saved model:", process.stdout)
            self.assertTrue(output_path.exists())

            artifact = load_network(output_path)
            self.assertEqual(
                artifact.training_config,
                AcceleratedTrainingConfig(
                    learning_rate=0.02,
                    max_power=1,
                    evaluation_points=64,
                    seed=5,
                    batch_size=32,
                    runtime=AcceleratedRuntime.numpy,
                ),
            )
            self.assertEqual(
                tuple(int(size) for size in artifact.network.config.hidden_layer_shapes),
                (4, 1),
            )
            self.assertEqual(
                artifact.network.config.layer_activation_funcs,
                (ActivationFunc.tanh, ActivationFunc.sigmoid),
            )

    def test_cli_default_output_path_creates_named_model_under_results(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_name = "default-run"
            configure_process = run_configure_model(
                temp_dir,
                run_name,
                "--runtime",
                "numpy",
                "--max-power",
                "0",
                "--batch-size",
                "16",
            )
            self.assertEqual(configure_process.returncode, 0, msg=configure_process.stderr)

            process = subprocess.run(
                [
                    sys.executable,
                    str(TRAIN_SCRIPT_PATH),
                    run_name,
                    "--dataset-path",
                    str(DATASET_PATH),
                ],
                cwd=temp_dir,
                capture_output=True,
                text=True,
            )

            self.assertEqual(process.returncode, 0, msg=process.stderr)
            model_path = Path(temp_dir) / "results" / run_name / "model.ffnnpy"
            self.assertTrue(model_path.exists())
            self.assertIn(str(model_path.resolve()), process.stdout)

    def test_cli_writes_training_history_sidecar(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_name = "history-run"
            configure_process = run_configure_model(
                temp_dir,
                run_name,
                "--runtime",
                "numpy",
                "--max-power",
                "2",
                "--batch-size",
                "16",
            )
            self.assertEqual(configure_process.returncode, 0, msg=configure_process.stderr)

            process = subprocess.run(
                [
                    sys.executable,
                    str(TRAIN_SCRIPT_PATH),
                    run_name,
                    "--dataset-path",
                    str(DATASET_PATH),
                ],
                cwd=temp_dir,
                capture_output=True,
                text=True,
            )

            self.assertEqual(process.returncode, 0, msg=process.stderr)
            history_path = Path(temp_dir) / "results" / run_name / "training_history.json"
            self.assertTrue(history_path.exists())
            payload = json.loads(history_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["source"], "recorded_during_training")
            self.assertEqual(payload["metric"], "evaluation_loss")
            self.assertEqual([point["step"] for point in payload["points"]], [1, 2, 4])
            self.assertAlmostEqual(payload["final_loss"], payload["points"][-1]["loss"])

    def test_saved_model_stats_replays_training_history_when_sidecar_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_name = "replay-run"
            configure_process = run_configure_model(
                temp_dir,
                run_name,
                "--runtime",
                "numpy",
                "--max-power",
                "1",
                "--batch-size",
                "16",
                "--seed",
                "5",
                "--split-seed",
                "7",
            )
            self.assertEqual(configure_process.returncode, 0, msg=configure_process.stderr)

            process = subprocess.run(
                [
                    sys.executable,
                    str(TRAIN_SCRIPT_PATH),
                    run_name,
                    "--dataset-path",
                    str(DATASET_PATH),
                ],
                cwd=temp_dir,
                capture_output=True,
                text=True,
            )

            self.assertEqual(process.returncode, 0, msg=process.stderr)
            model_path = Path(temp_dir) / "results" / run_name / "model.ffnnpy"
            history_path = Path(temp_dir) / "results" / run_name / "training_history.json"
            history_path.unlink()

            stats = self.stats_module.evaluate_saved_model(
                model_path=model_path,
                dataset_path=DATASET_PATH,
            )

            self.assertIn("training_history", stats)
            training_history = stats["training_history"]
            self.assertEqual(training_history["source"], "replayed_from_hyperparams")
            self.assertEqual([point["step"] for point in training_history["points"]], [1, 2])
            self.assertAlmostEqual(training_history["final_loss"], training_history["points"][-1]["loss"])
            self.assertAlmostEqual(
                training_history["verification"]["loss_delta_vs_saved_model"],
                0.0,
                places=12,
            )
            self.assertTrue(history_path.exists())

    def test_cli_progress_flag_prints_training_milestones(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_name = "progress-run"
            configure_process = run_configure_model(
                temp_dir,
                run_name,
                "--runtime",
                "numpy",
                "--max-power",
                "1",
                "--batch-size",
                "16",
            )
            self.assertEqual(configure_process.returncode, 0, msg=configure_process.stderr)

            process = subprocess.run(
                [
                    sys.executable,
                    str(TRAIN_SCRIPT_PATH),
                    run_name,
                    "--dataset-path",
                    str(DATASET_PATH),
                    "--progress",
                ],
                cwd=temp_dir,
                capture_output=True,
                text=True,
            )

            self.assertEqual(process.returncode, 0, msg=process.stderr)
            self.assertIn("training start:", process.stdout)
            self.assertIn("progress= 50.00%", process.stdout)
            self.assertIn("progress=100.00%", process.stdout)


if __name__ == "__main__":
    unittest.main()
