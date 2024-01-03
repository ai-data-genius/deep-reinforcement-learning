from collections import deque
from random import sample as rand_sample
from typing import List


class ReplayBuffer:
    def __init__(self: "ReplayBuffer", capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)

    def __len__(self: "ReplayBuffer") -> int:
        return len(self.buffer)

    def push(
        self: "ReplayBuffer",
        state: List[int],
        action: int,
        reward: int,
        next_state: List[int],
        done: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self: "ReplayBuffer", batch_size: int) -> list:
        return rand_sample(self.buffer, batch_size)
