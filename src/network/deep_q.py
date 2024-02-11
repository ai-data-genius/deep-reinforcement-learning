from typing import Any

from torch.nn import Linear, ReLU

from src.network import Network


class DeepQNet(Network):
    def __init__(
        self: "DeepQNet",
        input_size: int,
        hidden_size: int,
        output_size: int,
    ) -> None:
        super().__init__()

        self.input_size: int = input_size
        self.hidden_size: int = hidden_size
        self.output_size: int = output_size

        self.init_network()

    def init_network(self: "DeepQNet") -> None:
        self.fc1 = Linear(self.input_size, self.hidden_size)  # Première couche cachée
        self.relu = ReLU()  # Fonction d'activation
        self.fc2 = Linear(self.hidden_size, self.hidden_size)  # Deuxième couche cachée
        self.fc3 = Linear(self.hidden_size, self.output_size)  # Couche de sortie

    def forward(self: "DeepQNet", x: Any) -> Any:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        return self.fc3(x)
