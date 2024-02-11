from random import choice, random
from typing import List, Tuple, Union

from torch import (
    argmax,
    float32,
    long,
    no_grad,
    tensor,
)

from src.agent.cant_stop import CantStop


class DeepQLearning(CantStop):
    def __init__(self: "DeepQLearning", **kwargs) -> None:
        super().__init__(**kwargs)

    def remember(
        self: "DeepQLearning",
        state: List[int],
        action: Tuple[int, int],
        reward: float,
        next_state: List[int],
        done: bool,
    ):
        # Stocker l'expérience dans le buffer de replay
        self.memory.add(state, action, reward, next_state, done)

        # Assurer que la taille du buffer de replay ne dépasse pas la capacité maximale
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)  # Supprimer l'expérience la plus ancienne si nécessaire

    def update(self: "DeepQLearning", *args, **kwargs) -> None:
        if len(self.memory) < self.batch_size:
            return  # Ne pas mettre à jour si le buffer de replay n'a pas assez d'expériences

        # Échantillonner un minibatch d'expériences du buffer de replay
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Convertir les listes en tensors PyTorch
        states = tensor(states, dtype=float32)
        next_states = tensor(next_states, dtype=float32)
        actions = tensor(actions, dtype=long)
        rewards = tensor(rewards, dtype=float32)
        dones = tensor(dones, dtype=float32)

        # Calculer les valeurs Q actuelles (Q-values pour les actions effectuées)
        current_q_values = self.model(states).gather(1, actions)

        # Calculer les valeurs Q cibles pour les actions actuelles
        next_q_values = self.model(next_states).detach().max(1)[0]
        expected_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # Calculer la perte (loss) et effectuer une mise à jour du gradient
        loss = self.criterion(current_q_values, expected_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def select_action(
        self: "DeepQLearning",
        state: List[int],
        possible_actions: List[Tuple[int, int]],
        num_columns: int,
    ) -> Union[Tuple[int, int], bool]:
        self._update_epsilon()

        if random() < self.epsilon:
            return choice(possible_actions), random() >= self.keep_playing_threshold

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
