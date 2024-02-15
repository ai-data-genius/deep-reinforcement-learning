from typing import Any

from torch import Tensor, sigmoid
from torch.nn import Linear
from torch.nn.functional import relu, softmax

from src.network import Network


class PPO(Network):
    def __init__(
        self: "PPO",
        input_size: int,
        hidden_size: int,
        output_size: int,
    ) -> None:
        super().__init__()

        self.input_size: int = input_size
        self.hidden_size: int = hidden_size
        self.output_size: int = output_size

        self.init_network()

    def init_network(self: "PPO") -> None:
        self.fc1: Linear = Linear(self.input_size, self.hidden_size)
        self.policy_head: Linear = Linear(self.hidden_size, self.output_size)
        self.value_head: Linear = Linear(self.hidden_size, 1)
        self.keep_playing_head = Linear(self.hidden_size, 1)

    def forward(self: "PPO", x: Any) -> Any:
        x: Tensor = relu(self.fc1(x))

        return (
            softmax(self.policy_head(x), dim=-1),
            self.value_head(x),
            sigmoid(self.keep_playing_head(x)),
        )
