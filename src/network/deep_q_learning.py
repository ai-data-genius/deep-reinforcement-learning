from torch.nn import Linear, ReLU

from network import Network


class DeepQLearning(Network):
    def __init__(
        self: "DeepQLearning",
        input_size: int,
        hidden_size: int,
        output_size: int,
    ) -> None:
        super(Network, self).__init__()

        self.fc1 = Linear(input_size, hidden_size)  # Première couche cachée
        self.relu = ReLU()  # Fonction d'activation
        self.fc2 = Linear(hidden_size, hidden_size)  # Deuxième couche cachée
        self.fc3 = Linear(hidden_size, output_size)  # Couche de sortie

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        return self.fc3(x)
