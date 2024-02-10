from random import choice
from typing import List, Tuple, Union

from src.agent.cant_stop import CantStop as Agent
from src.entity.cant_stop.player import Player
from src.env.cant_stop import CantStop as Env


class RandomRollout(Agent):
    def __init__(self: "RandomRollout", num_rollouts: int = 10, rollout_depth: int = 3):
        super().__init__()

        self.num_rollouts = num_rollouts
        self.rollout_depth = rollout_depth

    def select_action(
        self: "RandomRollout",
        state: List[int],
        possible_actions: List[Tuple[int, int]],
        num_columns: int,
    ) -> Union[Tuple[int, int], bool]:
        # Choix d'une action aléatoire pour compatibilité avec l'interface
        return choice(possible_actions), True

    def rollout(self: "RandomRollout", env: Env, player: Player) -> float:
        """Effectue des rollouts aléatoires pour évaluer un état."""

        total_reward = 0
        for _ in range(self.num_rollouts):
            current_state = env.get_state()
            cumulative_reward = 0

            for _ in range(self.rollout_depth):
                possible_actions = env.get_possible_actions(player)
                action, _ = self.select_action(current_state, possible_actions, env.nb_ways)
                reward = env.step(current_state, action)
                cumulative_reward += reward

                if env.is_over:
                    break

                current_state = env.get_state()

            total_reward += cumulative_reward

        return total_reward / self.num_rollouts
