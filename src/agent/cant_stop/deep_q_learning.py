from random import choice, random
from typing import List, Tuple, Union

from torch import (
    argmax,
    float32,
    no_grad,
    tensor,
)

from agent.cant_stop import CantStop


class DeepQLearning(CantStop):
    def __init__(self: "DeepQLearning", **kwargs) -> None:
        super(DeepQLearning, self).__init__(**kwargs)

    def _action_to_index(
        self: "DeepQLearning",
        action: Tuple[int],
        num_columns: int,
    ) -> int:
        # -2 pour l'ajustement pour le décalage de numérotation (ex: 2 à 12 au lieu de 1 à 11)
        return (
            (action[0] - 2)
            * num_columns
            + (
                (action[1] - 1)
                if len(action) == 2
                else (action[0] - 2)
            )
        )

    def select_action(
        self: "DeepQLearning",
        state: List[int],
        possible_actions: List[Tuple[int]],
        num_columns: int,
    ) -> Union[Tuple[int], bool]:
        if random() < self.epsilon:
            self.update_epsilon()

            return choice(possible_actions), random() >= self.keep_playing_threshold

        self.update_epsilon()

        with no_grad():
            # Récupération de toutes les valeurs Q
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

    def update_epsilon(self: "DeepQLearning") -> None:
        self.epsilon = max(self.epsilon * self.decay_rate, 0.01)  # Limite minimum à 0.01
