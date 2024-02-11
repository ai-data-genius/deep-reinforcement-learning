from random import random, choice
from typing import List, Tuple, Union

from torch import (
    argmax,
    float32,
    long,
    no_grad,
    tensor,
    zeros
)

from src.agent.cant_stop import CantStop


class DoubleDeepQLearning(CantStop):
    def __init__(self, target_update_frequency, **kwargs) -> None:
        super().__init__(**kwargs)

        # Initialiser le réseau cible avec les mêmes poids que le modèle (réseau d'évaluation)
        self.target_network = type(self.model)(
            **{
                k: getattr(kwargs["model"], k)
                for k in dir(kwargs["model"])
                if k in ["input_size", "hidden_size", "output_size"]
            }
        )

        self.target_network.load_state_dict(self.model.state_dict())
        self.target_network.eval()  # Mettre le réseau cible en mode évaluation
        self.target_update_frequency = target_update_frequency
        self.update_count = 0

    def update(
        self: "DoubleDeepQLearning",
        state: List[int],
        action: Tuple[int, int],
        reward: float,
        next_state: List[int],
        done: bool,
    ) -> None:
        state = tensor(state, dtype=float32)
        action = tensor(action, dtype=long)
        reward = tensor(reward, dtype=float32)
        next_state = tensor(next_state, dtype=float32)
        done = tensor(done, dtype=float32)

        state_action_values = self.model(state)

        next_state_values = zeros(1)
        if not done:
            next_state_values = self.target_network(next_state).max(0)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma) + reward

        loss = self.criterion(state_action_values, expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Mise à jour périodique du réseau cible
        self.update_count += 1
        if self.update_count % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.model.state_dict())

    def select_action(
        self: "DoubleDeepQLearning",
        state: List[int],
        possible_actions: List[Tuple[int, int]],
        num_columns: int,
    ) -> Union[Tuple[int, int], bool]:
        self._update_epsilon()

        if random() < self.epsilon:
            return choice(possible_actions), random() >= self.keep_playing_threshold

        with no_grad():
            all_q_values = self.model(tensor(state, dtype=float32))

        return (
            possible_actions[
                argmax(
                    tensor(
                        [
                            all_q_values[self._action_to_index(action, num_columns)]
                            for action in possible_actions
                        ]
                    )
                )
                .item()
            ],
            all_q_values[-1].item() >= self.keep_playing_threshold,
        )
