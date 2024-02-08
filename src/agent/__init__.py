from typing import Callable, List

from src.agent.tool.replay_buffer import ReplayBuffer


class Agent:
    batch_size: int
    criterion: Callable
    gamma: float
    memory: ReplayBuffer
    optimizer: object

    def select_action(self: "Agent", state: List[int]) -> int:
        raise NotImplementedError

    def train(self: "Agent") -> int:
        raise NotImplementedError
