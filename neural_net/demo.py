import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

from .accelerated import (
    AcceleratedRuntime,
    AcceleratedTrainingConfig,
    build_accelerated_network,
    fit_function_accelerated,
)
from .backend import ActivationFunc
from .training import AsyncProgressPrinter, TrainingConfig, build_random_network, fit_function


DEFAULT_SAVE_PATH = "sin_learning_milestones.png"


def plot_training_snapshots(
    xs: np.ndarray,
    ys_true: np.ndarray,
    snapshots: dict[int, np.ndarray],
    losses: dict[int, float],
    *,
    save_path: str | Path = DEFAULT_SAVE_PATH,
    title_suffix: str = "",
):
    milestones = sorted(snapshots)
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    fig.subplots_adjust(bottom=0.22)

    ax.plot(xs, ys_true, color="#0f172a", linewidth=2.2, label="target f(x)")
    initial_milestone = milestones[0]
    line_pred, = ax.plot(
        xs,
        snapshots[initial_milestone].reshape(-1),
        color="#2563eb",
        linewidth=2.2,
        label="network",
    )
    ax.set_xlim(xs[0], xs[-1])
    y_min = min(float(np.min(ys_true)), *(float(np.min(pred)) for pred in snapshots.values()))
    y_max = max(float(np.max(ys_true)), *(float(np.max(pred)) for pred in snapshots.values()))
    y_pad = 0.1 * max(1.0, y_max - y_min)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.grid(alpha=0.25)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right")

    title = ax.set_title(
        f"Learning sin(x) on [-pi, pi]\n"
        f"n = 0, updates = {initial_milestone}, MSE = {losses[initial_milestone]:.4f}{title_suffix}"
    )

    slider_ax = fig.add_axes((0.14, 0.08, 0.72, 0.04))
    slider = Slider(
        ax=slider_ax,
        label="n",
        valmin=0,
        valmax=len(milestones) - 1,
        valinit=0,
        valstep=np.arange(len(milestones)),
        valfmt="%0.0f",
    )

    def _update(snapshot_index):
        snapshot_index = int(snapshot_index)
        milestone = milestones[snapshot_index]
        line_pred.set_ydata(snapshots[milestone].reshape(-1))
        title.set_text(
            f"Learning sin(x) on [-pi, pi]\n"
            f"n = {snapshot_index}, updates = {milestone}, MSE = {losses[milestone]:.4f}{title_suffix}"
        )
        fig.canvas.draw_idle()

    slider.on_changed(_update)

    if save_path is not None:
        output_path = Path(save_path)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Saved plot to {output_path.resolve()}")

    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a small FFNN to learn sin(x) and visualize milestone predictions."
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="print milestone training statistics without blocking the hot training loop",
    )
    parser.add_argument(
        "-n",
        "--runs",
        type=int,
        default=15,
        help="set number of training passes as a power of 2. e.g. n = 6 -> 2^6 = 64 training runs"
    )
    parser.add_argument(
        "--backend",
        choices=("reference", "accelerated"),
        default="reference",
        help="choose the reference scalar backend or the accelerated batched backend",
    )
    parser.add_argument(
        "--runtime",
        choices=tuple(runtime.value for runtime in AcceleratedRuntime),
        default=AcceleratedRuntime.auto.value,
        help="runtime for the accelerated backend",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="mini-batch size for the accelerated backend",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    title_suffix = ""

    with AsyncProgressPrinter(enabled=args.verbose) as progress_printer:
        if args.backend == "accelerated":
            config = AcceleratedTrainingConfig(
                max_power=args.runs,
                batch_size=args.batch_size,
                runtime=AcceleratedRuntime(args.runtime),
            )
            network = build_accelerated_network(
                input_layer_dim=1,
                hidden_layer_shapes=(32, 32, 1),
                activation=ActivationFunc.tanh,
                seed=config.seed,
                runtime=config.runtime,
            )
            result = fit_function_accelerated(
                network,
                np.sin,
                config=config,
                progress_logger=progress_printer.log if args.verbose else None,
            )
            resolved_runtime = network.resolve_runtime(config.runtime)
            title_suffix = (
                f"\nbackend=accelerated, runtime={resolved_runtime.value}, "
                f"batch={config.batch_size}, samples_seen={config.batch_size * max(result.snapshots)}"
            )
        else:
            config = TrainingConfig(max_power=args.runs)
            network = build_random_network(
                input_layer_dim=1,
                hidden_layer_shapes=(32, 32, 1),
                activation=ActivationFunc.tanh,
                seed=config.seed,
            )
            result = fit_function(
                network,
                np.sin,
                config=config,
                progress_logger=progress_printer.log if args.verbose else None,
            )
            title_suffix = "\nbackend=reference"

    plot_training_snapshots(
        result.evaluation_inputs.reshape(-1),
        result.evaluation_targets.reshape(-1),
        result.snapshots,
        result.losses,
        title_suffix=title_suffix,
    )


if __name__ == "__main__":
    main()
