from neural_network.activations import relu
from neural_network.numerics import _shape, matvec
from typing import Callable

class Dense:
    def __init__(
        self,
        weights: list[list[float]],
        biases: list[float],
        activation: Callable[[float], float]
    ):
        """Initialize a fully connected neural network layer."""
        # Validate dimensions and store parameters.
        m, n = _shape(weights)
        if len(biases) != m:
            raise ValueError(f'Mismatched bias count {len(biases)} and neurons {m}')

        self._weights = [row[:] for row in weights]
        self._biases = biases[:]
        self.activation_function = activation
        self._input_size = n
        self._output_size = m

    @property
    def input_size(self) -> int:
        return self._input_size

    @property
    def output_size(self) -> int:
        return self._output_size

    def parameter_count(self) -> int:
        return self._output_size * (self._input_size + 1)

    def forward(
        self,
        inputs: list[float]
    ) -> list[float]:
        """Compute the activated output of the layer."""
        if len(inputs) != self.input_size:
            raise ValueError(f'Expected input length {self.input_size}, got {len(inputs)}')
        matrix_product = matvec(self._weights, inputs)

        # add biases
        n = len(matrix_product)
        bias_count = len(self._biases)
        if bias_count != n:
            raise ValueError(f'Mismatched dimensions between biases {bias_count} and vector {n}')
        for i in range(n):
            matrix_product[i] += self._biases[i]

        # calcuate ReLU
        for i in range(n):
            matrix_product[i] = self.activation_function(matrix_product[i])
        return matrix_product