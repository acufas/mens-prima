from neural_network import Dense

class Sequential:
    def __init__(self, layers: list[Dense]):
        # Store layers.
        # Validate architectural compatibility.
        raise NotImplementedError

    def forward(
        self,
        inputs: list[float]
    ) -> list[float]:
        raise NotImplementedError

    def parameter_count(self) -> int:
        raise NotImplementedError
