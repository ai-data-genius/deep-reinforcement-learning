from typing import Any

from torch import Tensor, sigmoid
from torch.nn import Linear
from torch.nn.functional import relu, softmax

from src.network import Network


class Reinforce(Network):
    def __init__(
        self: "Reinforce",
        input_size: int,
        hidden_size: int,
        output_size: int,
    ) -> None:
        super().__init__()

        self.input_size: int = input_size
        self.hidden_size: int = hidden_size
        self.output_size: int = output_size

        self.init_network()

    def init_network(self: "Reinforce") -> None:
        self.fc1 = Linear(self.input_size, self.hidden_size)
        self.fc2 = Linear(self.hidden_size, self.hidden_size)
        self.action_head = Linear(self.hidden_size, self.output_size)
        self.continue_playing_head = Linear(self.hidden_size, 1)

    def forward(self: "Reinforce", x: Any) -> Any:
        x: Tensor = relu(self.fc1(x))
        x: Tensor = relu(self.fc2(x))
        action_probs = softmax(self.action_head(x), dim=-1)
        continue_playing_prob = sigmoid(self.continue_playing_head(x))

        return action_probs, continue_playing_prob
