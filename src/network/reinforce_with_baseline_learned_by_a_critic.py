from typing import Any

from torch import Tensor, sigmoid
from torch.nn import Linear
from torch.nn.functional import relu, softmax

from src.network import Network


class ActorCritic(Network):
    def __init__(
        self: "ActorCritic",
        input_size: int,
        hidden_size: int,
        output_size: int,
    ) -> None:
        super().__init__()

        self.input_size: int = input_size
        self.hidden_size: int = hidden_size
        self.output_size: int = output_size

        self.init_network()

    def init_network(self: "ActorCritic") -> None:
        self.affine: Linear = Linear(self.input_size, self.hidden_size)
        self.action_head: Linear = Linear(self.hidden_size, self.output_size)
        self.value_head: Linear = Linear(self.hidden_size, 1)
        self.continue_playing_head = Linear(self.hidden_size, 1)

    def forward(self: "ActorCritic", x: Any) -> Any:
        x: Tensor = relu(self.affine(x))
        action_probs: Tensor = softmax(self.action_head(x), dim=-1)
        state_values = self.value_head(x)
        continue_playing_prob = sigmoid(self.continue_playing_head(x))

        return action_probs, state_values, continue_playing_prob
