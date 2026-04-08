import numpy as np

from enum import Enum
from typing import Any, Callable, Sequence

OutputModifier = Callable[[np.ndarray], Any]

class ActivationFunc(Enum):
    relu = "relu"
    tanh = "tanh"
    sigmoid = "sigmoid"
    inv_quad = "inv_quad"

class LossFunc(Enum):
    mse = "mse"


def _coerce_activation_func(func: ActivationFunc | str) -> ActivationFunc:
    if isinstance(func, ActivationFunc):
        return func

    try:
        return ActivationFunc(func)
    except ValueError as exc:
        raise ValueError(f"Function {func} not supported") from exc


def _normalize_activation_funcs(
    activation_func: ActivationFunc | str | Sequence[ActivationFunc | str],
    hidden_layer_count: int,
) -> tuple[ActivationFunc, ...]:
    if isinstance(activation_func, (ActivationFunc, str)):
        activation = _coerce_activation_func(activation_func)
        return tuple(activation for _ in range(hidden_layer_count))

    activation_funcs = tuple(_coerce_activation_func(func) for func in activation_func)
    if len(activation_funcs) != hidden_layer_count:
        raise ValueError(
            "activation_func sequence length must match hidden_layer_count"
        )

    return activation_funcs


class FFNNConfig:
    input_layer_dim: int = 1
    hidden_layer_count: int = 3
    hidden_layer_shapes: np.ndarray = np.ones(hidden_layer_count, dtype=int)
    activation_func: ActivationFunc | tuple[ActivationFunc, ...] = ActivationFunc.tanh
    layer_activation_funcs: tuple[ActivationFunc, ...] = (ActivationFunc.tanh,)
    loss_func: LossFunc = LossFunc.mse
    output_modifier: OutputModifier | None = None

    def __init__(
        self,
        input_layer_dim: int = 1,
        hidden_layer_count: int = 3,
        hidden_layer_shapes: np.ndarray | list[int] | tuple[int, ...] | None = None,
        activation_func: ActivationFunc | str | Sequence[ActivationFunc | str] = ActivationFunc.tanh,
        loss_func: LossFunc = LossFunc.mse,
        output_modifier: OutputModifier | None = None,
    ):
        self.input_layer_dim = int(input_layer_dim)
        self.hidden_layer_count = int(hidden_layer_count)
        if self.hidden_layer_count < 1:
            raise ValueError("hidden_layer_count must be at least 1")

        if hidden_layer_shapes is None:
            self.hidden_layer_shapes = np.ones(self.hidden_layer_count, dtype=int)
        else:
            self.hidden_layer_shapes = np.asarray(hidden_layer_shapes, dtype=int).reshape(-1)
            if self.hidden_layer_shapes.shape[0] != self.hidden_layer_count:
                raise ValueError(
                    "hidden_layer_count must match the number of hidden_layer_shapes entries"
                )
            if np.any(self.hidden_layer_shapes < 1):
                raise ValueError("hidden_layer_shapes entries must be at least 1")

        self.layer_activation_funcs = _normalize_activation_funcs(
            activation_func,
            self.hidden_layer_count,
        )
        if isinstance(activation_func, (ActivationFunc, str)):
            self.activation_func = self.layer_activation_funcs[0]
        else:
            self.activation_func = self.layer_activation_funcs
        self.loss_func = loss_func
        self.output_modifier = output_modifier


def get_loss_func(func: LossFunc):
    if func.value == "mse":
        def mse(y: np.ndarray, y_hat: np.ndarray):
            y = np.asarray(y, dtype=float).reshape(-1)
            y_hat = np.asarray(y_hat, dtype=float).reshape(-1)
            if y.shape != y_hat.shape:
                raise ValueError("y and y_hat must have the same shape")
            if y.size == 0:
                raise ValueError("y and y_hat must be non-empty")
            return np.mean((y - y_hat) ** 2)
        return mse
    raise ValueError(f"Loss function {func} not supported")


def get_loss_func_derivative(func: LossFunc):
    if func.value == "mse":
        def mse_derivative(y: np.ndarray, y_hat: np.ndarray):
            y = np.asarray(y, dtype=float).reshape(-1)
            y_hat = np.asarray(y_hat, dtype=float).reshape(-1)
            if y.shape != y_hat.shape:
                raise ValueError("y and y_hat must have the same shape")
            if y.size == 0:
                raise ValueError("y and y_hat must be non-empty")
            return 2 * (y_hat - y) / y.size

        return mse_derivative

    raise ValueError(f"Loss function {func} not supported")


def get_activation(func: ActivationFunc):
    func = _coerce_activation_func(func)
    if func.value == "relu":
        def relu(x: np.ndarray):
            return np.maximum(x, 0)
        
        return relu
    
    elif func.value == "tanh":
        def tanh(x: np.ndarray):
            return np.tanh(x)
        
        return tanh
    
    elif func.value == "sigmoid":
        def sigmoid(x: np.ndarray):
            return 1 / (np.exp(-x) + 1)
        
        return sigmoid
    
    elif func.value == "inv_quad":
        def inv_quad(x: np.ndarray):
            return 1 / (1 + x**2)
        
        return inv_quad
    
    raise ValueError(f"Function {func} not supported")

def get_activation_derivative(func: ActivationFunc):
    func = _coerce_activation_func(func)
    if func.value == "relu":
        def relu_derivative(x: np.ndarray):
            return (np.asarray(x) > 0).astype(float)
        
        return relu_derivative
    
    elif func.value == "tanh":
        def tanh_derivative(x: np.ndarray):
            return 1 - np.tanh(x)**2
        
        return tanh_derivative
    
    elif func.value == "sigmoid":
        def sigmoid_derivative(x: np.ndarray):
            sigma = 1 / (np.exp(-x) + 1)
            return sigma * (1 - sigma)
        
        return sigmoid_derivative
    
    elif func.value == "inv_quad":
        def inv_quad_derivative(x: np.ndarray):
            return -2 * x / (1 + x**2)**2
        
        return inv_quad_derivative

    raise ValueError(f"Function {func} not supported")


def _apply_output_modifier(
    output_modifier: OutputModifier | None,
    raw_output: np.ndarray,
):
    sample_output = np.array(raw_output, dtype=float, copy=True).reshape(-1)
    if output_modifier is None:
        return sample_output
    return output_modifier(sample_output)


def _apply_output_modifier_batch(
    raw_outputs: np.ndarray,
    output_modifier: OutputModifier | None,
) -> np.ndarray:
    output_rows = np.asarray(raw_outputs, dtype=float)
    if output_rows.ndim == 1:
        output_rows = output_rows.reshape(1, -1)

    if output_modifier is None:
        return output_rows

    modified_outputs = [
        np.asarray(_apply_output_modifier(output_modifier, sample_output))
        for sample_output in output_rows
    ]
    expected_shape = modified_outputs[0].shape if modified_outputs else ()

    for modified_output in modified_outputs[1:]:
        if modified_output.shape != expected_shape:
            raise ValueError(
                "output_modifier must return values with a consistent shape for each sample"
            )

    if expected_shape == ():
        return np.asarray([modified_output.item() for modified_output in modified_outputs])

    return np.stack(modified_outputs, axis=0)


class FFNN():
    def __init__(self, config: FFNNConfig):
        self.config = config
        self.output_modifier = self.config.output_modifier

        self.pre_activations: np.ndarray = np.empty(self.config.hidden_layer_count, dtype = np.ndarray)
        self.values: np.ndarray          = np.empty(self.config.hidden_layer_count, dtype=np.ndarray)
        self.biases: np.ndarray          = np.empty(self.config.hidden_layer_count, dtype=np.ndarray)
        for i in range(self.values.shape[0]):
            self.pre_activations[i] = np.zeros(self.config.hidden_layer_shapes[i])
            self.values[i]          = np.zeros(self.config.hidden_layer_shapes[i])
            self.biases[i]          = np.random.random((self.config.hidden_layer_shapes[i]))

        self.weights: np.ndarray = np.empty(self.config.hidden_layer_count, dtype=np.ndarray)
        for i in range(self.weights.shape[0]):
            self.weights[i] = np.random.random((self.config.hidden_layer_shapes[i], self.config.hidden_layer_shapes[i-1] if i-1 >= 0 else self.config.input_layer_dim)) #shape = (out, in)

        self.activations = tuple(
            get_activation(func) for func in self.config.layer_activation_funcs
        )
        self.activation_derivatives = tuple(
            get_activation_derivative(func) for func in self.config.layer_activation_funcs
        )
        self.activation = (
            self.activations[0]
            if len(set(self.config.layer_activation_funcs)) == 1
            else self.activations
        )

    def _raw_forward_pass(self, x_: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
        x = np.asarray(x_, dtype=np.float64).reshape(-1)
        if x.shape[0] != self.config.input_layer_dim:
            raise ValueError("input shape must match input layer shape")
        
        for i in range(self.config.hidden_layer_count):
            self.pre_activations[i] = self.weights[i] @ (self.values[i-1] if i-1 >= 0 else x) + self.biases[i]
            self.values[i] = self.activations[i](self.pre_activations[i])
            
        return self.values[-1]

    def fast_forward_pass(self, x_: np.ndarray | list[float] | tuple[float, ...]):
        return _apply_output_modifier(self.output_modifier, self._raw_forward_pass(x_))
    
    def fast_backward_pass(self, x: np.ndarray, y_actual: np.ndarray, learning_rate: float):
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        y_pred = np.asarray(self._raw_forward_pass(x), dtype=np.float64).reshape(-1)
        y_actual = np.asarray(y_actual, dtype=np.float64).reshape(-1)
        if y_actual.shape != y_pred.shape:
            raise ValueError("y_actual shape must match the network output shape")

        #last layer
        loss_deriv = get_loss_func_derivative(self.config.loss_func)

        deltas = np.empty(self.config.hidden_layer_count, dtype=np.ndarray)
        grad_weights = np.empty(self.config.hidden_layer_count, dtype=np.ndarray)
        grad_biases = np.empty(self.config.hidden_layer_count, dtype=np.ndarray)

        deltas[-1] = loss_deriv(y_actual, y_pred) * self.activation_derivatives[-1](self.pre_activations[-1])
        prev_activations = self.values[-2] if self.config.hidden_layer_count > 1 else x
        grad_weights[-1] = np.outer(deltas[-1], prev_activations)
        grad_biases[-1] = deltas[-1]

        for i in range(self.config.hidden_layer_count - 2, -1, -1):
            deltas[i] = (
                self.weights[i + 1].T @ deltas[i + 1]
            ) * self.activation_derivatives[i](self.pre_activations[i])
            prev_activations = self.values[i - 1] if i - 1 >= 0 else x
            grad_weights[i] = np.outer(deltas[i], prev_activations)
            grad_biases[i] = deltas[i]

        for i in range(self.config.hidden_layer_count):
            self.weights[i] -= grad_weights[i] * learning_rate
            self.biases[i] -= grad_biases[i] * learning_rate


    def pretty_print_network(
        self,
        input_layer: np.ndarray | None = None,
        *,
        decimals: int = 3,
        draw: bool = True,
        save_path: str | None = None,
    ):
        hidden_layer_shapes = tuple(
            int(size) for size in np.atleast_1d(self.config.hidden_layer_shapes).tolist()
        )

        if input_layer is None:
            input_layer = np.zeros(self.config.input_layer_dim, dtype=float)
        else:
            input_layer = np.reshape(
                np.asarray(input_layer, dtype=float),
                (self.config.input_layer_dim,),
            )

        layer_values = [
            np.asarray(layer, dtype=float).reshape(-1)
            for layer in np.atleast_1d(np.asarray(self.values, dtype=object))
        ]
        layer_biases = [
            np.asarray(layer, dtype=float).reshape(-1)
            for layer in np.atleast_1d(np.asarray(self.biases, dtype=object))
        ]
        layer_weights = []
        for layer in np.atleast_1d(np.asarray(self.weights, dtype=object)):
            weights = np.asarray(layer, dtype=float)
            if weights.ndim == 0:
                weights = weights.reshape(1, 1)
            elif weights.ndim == 1:
                weights = weights.reshape(1, -1)
            layer_weights.append(weights)

        def _vector_text(values, limit: int = 6):
            flat = np.asarray(values, dtype=float).reshape(-1)
            shown = ", ".join(f"{value:.{decimals}f}" for value in flat[:limit])
            if flat.size > limit:
                shown += ", ..."
            return f"[{shown}]"

        def _activation_text():
            activation_names = [func.value for func in self.config.layer_activation_funcs]
            if len(set(self.config.layer_activation_funcs)) != 1:
                return ", ".join(
                    f"L{layer_index + 1}={activation_name}"
                    for layer_index, activation_name in enumerate(activation_names)
                )

            activation_name = activation_names[0]
            if not callable(getattr(self, "activation", None)):
                return activation_name

            sample_points = np.array([0.0, 0.5, 1.0])
            try:
                sample_values = np.asarray(self.activation(sample_points), dtype=float).reshape(-1)
            except Exception:
                return activation_name

            return f"{activation_name} -> {_vector_text(sample_values, limit=3)}"

        layer_sizes = [self.config.input_layer_dim, *hidden_layer_shapes]

        print("Network summary")
        print("=" * 72)
        print(f"Shape          : {' -> '.join(str(size) for size in layer_sizes)}")
        print(f"Input values   : {_vector_text(input_layer, limit=self.config.input_layer_dim)}")
        print(f"Activation     : {_activation_text()}")
        print(f"Weight tensors : {[tuple(weights.shape) for weights in layer_weights]}")
        print(f"Bias vectors   : {[tuple(biases.shape) for biases in layer_biases]}")
        print()

        for layer_index, layer_size in enumerate(hidden_layer_shapes):
            values = layer_values[layer_index]
            biases = layer_biases[layer_index]
            weights = layer_weights[layer_index]

            print(f"Layer {layer_index + 1}")
            print(f"  values : {_vector_text(values, limit=layer_size)}")
            print(f"  biases : {_vector_text(biases, limit=layer_size)}")
            for node_index in range(min(layer_size, weights.shape[0])):
                print(
                    f"  node {node_index + 1} : "
                    f"w={_vector_text(weights[node_index], limit=weights.shape[1])}"
                )
            print()

        if not draw:
            return layer_sizes

        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError:
            print("Diagram skipped: install matplotlib to render the network inline.")
            return layer_sizes

        def _node_label(layer_index: int | None, node_index: int):
            if layer_index is None:
                return f"Input\nnode {node_index + 1}\nv={input_layer[node_index]:.{decimals}f}"

            values = layer_values[layer_index]
            biases = layer_biases[layer_index]
            weights = layer_weights[layer_index]
            return (
                f"Layer {layer_index + 1}\nnode {node_index + 1}\n"
                f"v={values[node_index]:.{decimals}f}\n"
                f"b={biases[node_index]:.{decimals}f}\n"
                f"w={_vector_text(weights[node_index], limit=4)}"
            )

        layer_names = ["Input", *[f"Layer {index + 1}" for index in range(len(hidden_layer_shapes))]]
        layer_colors = ["#dbeafe", *["#fef3c7" for _ in hidden_layer_shapes]]

        fig, ax = plt.subplots(
            figsize=(max(8, 2.4 * len(layer_sizes)), max(4.5, 1.25 * max(layer_sizes)))
        )

        layer_positions = []
        x_positions = np.linspace(0.08, 0.92, len(layer_sizes))
        for x_pos, size in zip(x_positions, layer_sizes):
            if size == 1:
                y_positions = np.array([0.5])
            else:
                y_positions = np.linspace(0.84, 0.16, size)
            layer_positions.append([(x_pos, y_pos) for y_pos in y_positions])

        for left_layer, right_layer in zip(layer_positions, layer_positions[1:]):
            for x0, y0 in left_layer:
                for x1, y1 in right_layer:
                    ax.plot(
                        [x0, x1],
                        [y0, y1],
                        color="#94a3b8",
                        linewidth=1.0,
                        alpha=0.45,
                        zorder=0,
                    )

        for layer_index, (layer_name, facecolor, positions) in enumerate(
            zip(layer_names, layer_colors, layer_positions)
        ):
            for node_index, (x_pos, y_pos) in enumerate(positions):
                ax.text(
                    x_pos,
                    y_pos,
                    _node_label(None, node_index) if layer_index == 0 else _node_label(layer_index - 1, node_index),
                    ha="center",
                    va="center",
                    fontsize=9,
                    bbox={
                        "boxstyle": "round,pad=0.45",
                        "facecolor": facecolor,
                        "edgecolor": "#0f172a",
                        "linewidth": 1.2,
                    },
                )

        ax.text(
            0.5,
            1.03,
            "Connections are schematic; each node box lists its stored value, bias, and weights.",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=9,
            color="#475569",
        )
        ax.text(
            0.5,
            -0.04,
            f"Shape: {' -> '.join(str(size) for size in layer_sizes)}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=11,
            color="#0f172a",
        )
        ax.set_axis_off()
        fig.tight_layout()

        if save_path is not None:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")

        plt.show()
        return layer_sizes
    
__all__ = [
    "ActivationFunc",
    "LossFunc",
    "FFNNConfig",
    "FFNN",
    "get_loss_func",
    "get_loss_func_derivative",
    "get_activation",
    "get_activation_derivative",
]
