from collections import deque
from random import sample

from torch import float as t_float, long, Tensor, tensor

from typing import List, Tuple


class PPOReplayBuffer:
    def __init__(
        self: "PPOReplayBuffer",
        buffer_size: int,
    ) -> None:
        self.buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.counter = 0

        # Initialiser les listes pour stocker les états, actions, etc.
        self.state_memory: List[List[int]] = []
        self.action_memory: List[Tuple[int, int]] = []
        self.reward_memory: List[float] = []
        self.next_state_memory: List[List[int]] = []
        self.done_memory: List[bool] = []
        self.log_prob_memory: List[Tensor] = []
        self.value_memory: List[Tensor] = []

    def __len__(self: "PPOReplayBuffer") -> int:
        return len(self.buffer)

    def add(
        self: "PPOReplayBuffer",
        state: List[int],
        action: Tuple[int, int],
        reward: float,
        next_state: List[int],
        done: bool,
        log_prob: Tensor,
        value: Tensor,
    ):
        if len(self.state_memory) < self.buffer_size:
            self.state_memory.append(state)
            self.action_memory.append(action)
            self.log_prob_memory.append(log_prob)
            self.reward_memory.append(reward)
            self.next_state_memory.append(next_state)
            self.done_memory.append(done)
            self.value_memory.append(value)
        else:
            # Remplace les anciennes expériences si le buffer est plein
            idx = self.counter % self.buffer_size
            self.state_memory[idx] = state
            self.action_memory[idx] = action
            self.log_prob_memory[idx] = log_prob
            self.reward_memory[idx] = reward
            self.next_state_memory[idx] = next_state
            self.done_memory[idx] = done
            self.value_memory[idx] = value

        self.counter += 1

    def sample(
        self: "PPOReplayBuffer",
        batch_size: int,
    ) -> Tuple[
        List[List[int]],
        List[Tuple[int, int]],
        List[float],
        List[List[int]],
        List[bool],
        List[Tensor],
        List[Tensor],
    ]:
        # S'assure que le batch size ne dépasse pas la taille du buffer
        batch_indices = sample(
            range(len(self.state_memory)),
            min(len(self.state_memory), batch_size),
        )

        # Utilise les indices pour récupérer les données du batch
        return (
            tensor([self.state_memory[i] for i in batch_indices], dtype=t_float),
            tensor([self.action_memory[i] for i in batch_indices], dtype=long),
            tensor([self.reward_memory[i] for i in batch_indices], dtype=t_float),
            tensor([self.next_state_memory[i] for i in batch_indices], dtype=t_float),
            tensor([self.done_memory[i] for i in batch_indices], dtype=t_float),
            tensor([self.log_prob_memory[i] for i in batch_indices], dtype=t_float),
            tensor([self.value_memory[i] for i in batch_indices], dtype=t_float),
        )
