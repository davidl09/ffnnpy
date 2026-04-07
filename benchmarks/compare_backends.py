from __future__ import annotations

import argparse
import math
import time

import numpy as np

from neural_net import (
    AcceleratedRuntime,
    ActivationFunc,
    build_accelerated_network,
    build_random_network,
    predict_dataset,
    predict_dataset_accelerated,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark the scalar reference backend against the accelerated backend."
    )
    parser.add_argument("--training-samples", type=int, default=65536)
    parser.add_argument("--evaluation-points", type=int, default=65536)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--min-speedup", type=float, default=10.0)
    parser.add_argument(
        "--runtime",
        choices=tuple(runtime.value for runtime in AcceleratedRuntime),
        default=AcceleratedRuntime.auto.value,
        help="runtime for the accelerated backend benchmark",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    runtime = AcceleratedRuntime(args.runtime)
    rng = np.random.default_rng(0)

    training_inputs = rng.uniform(
        -np.pi,
        np.pi,
        size=(args.training_samples, 1),
    ).astype(np.float64)
    training_targets = np.sin(training_inputs)
    evaluation_inputs = np.linspace(
        -np.pi,
        np.pi,
        args.evaluation_points,
        dtype=np.float64,
    ).reshape(-1, 1)

    reference_network = build_random_network(
        input_layer_dim=1,
        hidden_layer_shapes=(32, 32, 1),
        activation=ActivationFunc.tanh,
        seed=0,
    )
    accelerated_network = build_accelerated_network(
        input_layer_dim=1,
        hidden_layer_shapes=(32, 32, 1),
        activation=ActivationFunc.tanh,
        seed=0,
        runtime=runtime,
    )
    resolved_runtime = accelerated_network.resolve_runtime(runtime)

    reference_start = time.perf_counter()
    for x_sample, y_sample in zip(training_inputs, training_targets):
        reference_network.fast_backward_pass(x_sample, y_sample, 0.02)
    reference_training_elapsed = time.perf_counter() - reference_start

    accelerated_start = time.perf_counter()
    for start in range(0, training_inputs.shape[0], args.batch_size):
        end = start + args.batch_size
        accelerated_network.train_batch(
            training_inputs[start:end],
            training_targets[start:end],
            0.02,
            runtime=resolved_runtime,
        )
    accelerated_training_elapsed = time.perf_counter() - accelerated_start

    reference_inference_start = time.perf_counter()
    predict_dataset(reference_network, evaluation_inputs)
    reference_inference_elapsed = time.perf_counter() - reference_inference_start

    accelerated_inference_start = time.perf_counter()
    predict_dataset_accelerated(accelerated_network, evaluation_inputs, runtime=resolved_runtime)
    accelerated_inference_elapsed = time.perf_counter() - accelerated_inference_start

    training_speedup = reference_training_elapsed / accelerated_training_elapsed
    inference_speedup = reference_inference_elapsed / accelerated_inference_elapsed
    overall_pass = (
        training_speedup >= args.min_speedup and inference_speedup >= args.min_speedup
    )

    print("Backend benchmark")
    print("=" * 72)
    print(f"runtime                : {resolved_runtime.value}")
    print(f"training_samples       : {args.training_samples}")
    print(f"evaluation_points      : {args.evaluation_points}")
    print(f"batch_size             : {args.batch_size}")
    print(f"reference_train_s      : {reference_training_elapsed:.4f}")
    print(f"accelerated_train_s    : {accelerated_training_elapsed:.4f}")
    print(f"training_speedup_x     : {training_speedup:.2f}")
    print(f"reference_infer_s      : {reference_inference_elapsed:.4f}")
    print(f"accelerated_infer_s    : {accelerated_inference_elapsed:.4f}")
    print(f"inference_speedup_x    : {inference_speedup:.2f}")
    print(f"min_speedup_target_x   : {args.min_speedup:.2f}")
    print(f"status                 : {'PASS' if overall_pass else 'FAIL'}")

    if not overall_pass:
        raise SystemExit(
            math.ceil(
                max(
                    args.min_speedup - training_speedup,
                    args.min_speedup - inference_speedup,
                    1.0,
                )
            )
        )


if __name__ == "__main__":
    main()
