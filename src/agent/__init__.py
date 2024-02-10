from typing import Any, Callable, List, Tuple

from src.agent.tool.replay_buffer import ReplayBuffer


class Agent:
    batch_size: int
    criterion: Callable
    gamma: float
    memory: ReplayBuffer
    optimizer: object

    def select_action(
        self: "Agent",
        state: List[int],
        possible_actions: List[Tuple[int, int]],
        num_columns: int,
    ) -> Any:
        raise NotImplementedError()
