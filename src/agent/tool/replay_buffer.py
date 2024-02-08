from collections import deque
from random import sample as rand_sample
from typing import List


class ReplayBuffer:
    def __init__(self: "ReplayBuffer", capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)

    def __len__(self: "ReplayBuffer") -> int:
        return len(self.buffer)

    def add(
        self: "ReplayBuffer",
        state: List[int],
        action: int,
        reward: int,
        next_state: List[int],
        done: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self: "ReplayBuffer", batch_size: int) -> tuple:
        experiences = rand_sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)

        return list(states), list(actions), list(rewards), list(next_states), list(dones)
