
from random import random, choice
from typing import List, Tuple, Union

from numpy import max as n_max
from torch import argmax, float32, int64, long, no_grad, tensor

from src.agent.cant_stop import CantStop
from src.agent.tool.prioritized_replay_buffer import PrioritizedReplayBuffer


class DoubleDeepQLearningWithPrioritizedReplay(CantStop):
    def __init__(
        self: "DoubleDeepQLearningWithPrioritizedReplay",
        target_update_frequency,
        replay_buffer_size: int,
        alpha: float = 0.6,  # Paramètre alpha pour le Prioritized Replay Buffer
        beta_start: float = 0.4,  # Paramètre beta initial pour le calcul des poids d'importance
        beta_frames: int = 10000,  # Nombre de frames sur lesquels beta sera augmenté linéairement vers 1
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

        # Initialisation du buffer de replay d'expérience priorisé
        self.memory = PrioritizedReplayBuffer(replay_buffer_size, alpha)
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 0

    def remember(
        self: "DoubleDeepQLearningWithPrioritizedReplay",
        state: List[int],
        action: Tuple[int, int],
        reward: float,
        next_state: List[int],
        done: bool,
    ) -> None:
        state_tensor = tensor([state], dtype=float32)
        next_state_tensor = tensor([next_state], dtype=float32)
        action_tensor = tensor([action], dtype=int64)
        reward_tensor = tensor([reward], dtype=float32)
        done_tensor = tensor([done], dtype=float32)

        # Calculer l'erreur temporelle delta (TD Error) pour définir la priorité
        current_q = self.model(state_tensor).gather(1, action_tensor)
        next_q = self.target_network(next_state_tensor).detach().max(1)[0]
        td_error = reward_tensor + self.gamma * next_q * (1 - done_tensor) - current_q

        # Utiliser .mean() pour obtenir un scalaire si td_error a plusieurs éléments
        priority = td_error.abs().mean().item()

        self.memory.add(state, action, reward, next_state, done, priority=priority)

        # Assurer que la taille du buffer de replay ne dépasse pas la capacité maximale
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)  # Supprimer l'expérience la plus ancienne si nécessaire

    def update(self: "DoubleDeepQLearningWithPrioritizedReplay", *args, **kwargs) -> None:
        self.frame += 1
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)

        # Vérifier si le buffer contient assez d'expériences pour un batch d'apprentissage
        if len(self.memory) < self.batch_size:
            return

        # Extraire un minibatch d'expériences avec leurs indices et poids
        (states, actions, rewards, next_states, dones), indices, weights = self.memory.sample(self.batch_size, beta)

        # Conversion des données en tensors PyTorch
        states = tensor(states, dtype=float32)
        actions = tensor(actions, dtype=long)
        rewards = tensor(rewards, dtype=float32)
        next_states = tensor(next_states, dtype=float32)
        dones = tensor(dones, dtype=float32)
        weights = tensor(weights, dtype=float32)

        # Mise à jour du modèle avec le minibatch échantillonné
        # Calcul des valeurs Q actuelles (Q estimées pour les actions choisies)
        current_q_values = self.model(states).gather(1, actions)

        # Calcul des valeurs Q futures estimées à partir du réseau cible pour les actions maximales
        next_q_values = self.target_network(next_states).detach().max(1)[0]
        expected_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # Calculer la perte en tenant compte des poids d'importance
        loss = (current_q_values - expected_q_values.unsqueeze(1)).pow(2) * weights.unsqueeze(1)
        prios = loss + 1e-5  # Éviter les priorités nulles
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Mise à jour des priorités dans le buffer
        self.memory.update_priorities(indices, n_max(prios.detach().squeeze().numpy(), axis=1))

        # Mise à jour périodique du réseau cible
        self.update_count += 1
        if self.update_count % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.model.state_dict())

    def select_action(
        self: "DoubleDeepQLearningWithPrioritizedReplay",
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

