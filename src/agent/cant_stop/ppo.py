from random import random
from typing import List, Tuple, Union

from torch import (
    clamp,
    exp,
    float as t_float,
    float32,
    min,
    softmax,
    tensor,
)
from torch.distributions import Categorical

from src.agent.cant_stop import CantStop as Agent


class PPO(Agent):
    def __init__(self: "PPO", K_epochs, eps_clip, **kwargs) -> None:
        super().__init__(**kwargs)

        self.K_epochs = K_epochs
        self.eps_clip = eps_clip

    def remember(
        self: "PPO",
        state: List[int],
        action: Tuple[int, int],
        reward: float,
        next_state: List[int],
        done: bool,
    ) -> None:
        self.memory.add(
            state,
            action,
            reward,
            next_state,
            done,
            self.log_prob,
            self.value,
        )

    def select_action(
        self: "PPO",
        state: List[int],
        possible_actions: List[Tuple[int, int]],
        num_columns: int,
    ) -> Union[Tuple[int, int], bool]:
        (
            action_probs,
            self.value,
            keep_playing_logits,
        ) = self.model(tensor(state, dtype=t_float).unsqueeze(0))
        action_probs = action_probs.squeeze()

        # Conversion des actions possibles en indices valides
        indices = [self._action_to_index(action, num_columns) for action in possible_actions]

        # Appliquer softmax sur les probabilités valides pour obtenir une distribution
        m = Categorical(softmax(action_probs[indices], dim=0))
        action_index = m.sample().item()

        # Sélectionne l'action correspondante à partir des actions possibles
        chosen_action = possible_actions[action_index]

        # Stocke le log prob de l'action choisie pour l'apprentissage
        self.log_prob = m.log_prob(tensor([action_index], dtype=float32))

        return (
            chosen_action,
            keep_playing_logits.item() > self.keep_playing_threshold,
        )

    def update(
        self: "PPO",
        *args,
        epochs: int = 4,
        clip_param: float = 0.2,
        **kwargs,
    ) -> None:
        # On s'assure que la mémoire a assez d'expériences pour un batch
        if len(self.memory) < self.batch_size:
            return

        # Prépare les données à partir de la mémoire pour l'entraînement
        (
            states,
            actions,
            returns,
            _,
            _,
            old_log_probs,
            advantages
        ) = self.memory.sample(self.batch_size)

        # Normalisation des avantages
        advantages = (
            (advantages - advantages.mean())
            / (advantages.std() + 1e-8)
        )

        for _ in range(epochs):
            # Obtient les log_probs actuels, les valeurs étatiques
            # pour les états dans le batch
            (
                log_probs,
                state_values,
                dist_entropy
            ) = self.model.evaluate(states, actions)

            # Calcul du ratio des probabilités
            ratios = exp(log_probs - old_log_probs.detach())

            # Objectif de la politique PPO avec clipping
            policy_loss = -min(
                ratios * advantages,
                clamp(
                    ratios,
                    1.0 - clip_param,
                    1.0 + clip_param,
                ) * advantages,
            ).mean()

            # Fonction de perte de la valeur
            value_loss = self.criterion(returns, state_values)

            # Calcul de la perte totale
            total_loss = (
                policy_loss
                + 0.5
                * value_loss
                - 0.01
                * dist_entropy
            )

            # Mise à jour des poids
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            self.cumulative_losses.append(value_loss.detach().numpy().item())
