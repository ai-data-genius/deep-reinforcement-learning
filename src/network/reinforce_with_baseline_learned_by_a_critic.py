from typing import Any

from torch import Tensor
from torch.nn import Linear
from torch.nn.functional import relu, softmax

from src.network import Network


class ActorCritic(Network):
    def __init__(
        self: "ActorCritic",
        input_size: int,
        output_size: int,
    ) -> None:
        super().__init__()

        self.input_size: int = input_size
        self.output_size: int = output_size

        self.init_network()

    def init_network(self: "ActorCritic") -> None:
        self.affine: Linear = Linear(self.input_size, 128)
        self.action_head: Linear = Linear(128, self.output_size)
        self.value_head: Linear = Linear(128, 1)

    def forward(self: "ActorCritic", x: Any) -> Any:
        x: Tensor = relu(self.affine(x))
        action_probs: Tensor = softmax(self.action_head(x), dim=-1)
        state_values = self.value_head(x)

        return action_probs, state_values
