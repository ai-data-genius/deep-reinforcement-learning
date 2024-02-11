from random import random, choice
from typing import List, Tuple, Union

from torch import (
    argmax,
    float32,
    long,
    no_grad,
    tensor,
)

from src.agent.cant_stop import CantStop
from src.agent.tool.replay_buffer import ReplayBuffer


class DoubleDeepQLearningWithExperienceReplay(CantStop):
    def __init__(
        self: "DoubleDeepQLearningWithExperienceReplay",
        target_update_frequency,
        replay_buffer_size,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Réseau cible pour l'estimation des valeurs Q futures
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

        # Initialisation du buffer de replay d'expérience
        self.memory = ReplayBuffer(replay_buffer_size)
        self.replay_buffer_size = replay_buffer_size

    def remember(
        self: "DoubleDeepQLearningWithExperienceReplay",
        state: List[int],
        action: Tuple[int, int],
        reward: float,
        next_state: List[int],
        done: bool,
    ) -> None:
        # Stocker l'expérience dans le buffer de replay
        self.memory.add(state, action, reward, next_state, done)

        # Assurer que la taille du buffer de replay ne dépasse pas la capacité maximale
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)  # Supprimer l'expérience la plus ancienne si nécessaire

    def update(self: "DoubleDeepQLearningWithExperienceReplay", *args, **kwargs) -> None:
        if len(self.memory) < self.batch_size:
            return

        # Extraire un minibatch d'expériences
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Convertir les listes en Tensors PyTorch
        states = tensor(states, dtype=float32)
        actions = tensor(actions, dtype=long)  # Ajoute une dimension pour gather
        rewards = tensor(rewards, dtype=float32)
        next_states = tensor(next_states, dtype=float32)
        dones = tensor(dones, dtype=float32)

        # Calcul des valeurs Q actuelles (Q estimées pour les actions choisies)
        current_q_values = self.model(states).gather(1, actions)

        # Calcul des valeurs Q futures estimées à partir du réseau cible pour les actions maximales
        next_q_values = self.target_network(next_states).detach().max(1)[0]
        expected_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # Mise à jour du réseau d'évaluation
        loss = self.criterion(current_q_values, expected_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Mise à jour périodique du réseau cible
        self.update_count += 1
        if self.update_count % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.model.state_dict())

    def select_action(
        self: "DoubleDeepQLearningWithExperienceReplay",
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
