from typing import Callable, List, Tuple, Union

from agent import Agent
from agent.tool.replay_buffer import ReplayBuffer
from network import Network


class CantStop(Agent):
    def __init__(
        self: "CantStop",
        batch_size: int,
        criterion: Callable,
        decay_rate: float,
        epsilon: float,
        gamma: float,
        model: Network,
        memory: ReplayBuffer,
        num_columns: int,
        optimizer: object,
    ):
        self.batch_size: int = batch_size
        self.criterion: Callable = criterion
        self.decay_rate: float = decay_rate
        self.epsilon: float = epsilon
        self.gamma: float = gamma
        self.keep_playing_threshold: float = .5
        self.model: Network = model
        self.memory: ReplayBuffer = memory
        self.num_columns: int = num_columns
        self.optimizer: object = optimizer(self.model.parameters(), lr=.01)

    def select_action(
        self: "CantStop",
        state: List[int],
        possible_actions: List[Tuple[int]],
        num_columns: int,
    ) -> Union[Tuple[int], bool]:
        raise NotImplementedError

    def update_epsilon(self: "CantStop") -> None:
        raise NotImplementedError
