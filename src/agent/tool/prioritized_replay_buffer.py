from numpy import float32, ndarray, zeros
from numpy.random import choice
from typing import Any, Iterable, Tuple, Union

from src.agent.tool.replay_buffer import ReplayBuffer


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(
        self: "PrioritizedReplayBuffer",
        capacity: int,
        alpha: float = 0.6,
    ) -> None:
        super().__init__(capacity)

        self.priorities = zeros((capacity), dtype=float32)
        self.alpha: float = alpha
        self.pos: int = 0
        self.capacity: int = capacity

    def add(
        self: "PrioritizedReplayBuffer",
        *args,
        priority: float = 1.0,
    ) -> None:
        super().add(*args)

        self.priorities[self.pos] = priority ** self.alpha
        self.pos: int = (self.pos + 1) % self.capacity

    def sample(
        self: "PrioritizedReplayBuffer",
        batch_size: int,
        beta: float = 0.4,
    ) -> Union[list, Tuple[Iterable, ndarray, Any]]:
        if len(self) == 0:
            return []

        # Ajuster pour calculer les probabilités uniquement pour les
        # parties remplies du buffer
        filled_size = min(len(self.buffer), self.capacity)
        probabilities = self.priorities[:filled_size] ** self.alpha
        probabilities /= probabilities.sum()

        # Échantillonnage des indices basés sur les probabilités
        indices = choice(filled_size, size=batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]

        # Calcul des poids d'importance
        weights = (filled_size * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        # S'assurer que les données retournées sont dans le format attendu
        return list(map(list, zip(*samples))), indices, weights

    def update_priorities(
        self: "PrioritizedReplayBuffer",
        indices: Iterable[int],
        priorities: Iterable[float],
    ) -> None:
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority ** self.alpha
