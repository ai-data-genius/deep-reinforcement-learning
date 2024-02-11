from random import choice, random
from typing import List, Tuple, Union

from torch import (
    float32,
    softmax,
    stack,
    Tensor,
    tensor,
)
from torch.distributions import Categorical
from torch.nn.functional import smooth_l1_loss

from src.agent.cant_stop import CantStop


class ReinforceWithBaselineLearnedByACritic(CantStop):
    def __init__(
        self: "ReinforceWithBaselineLearnedByACritic",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.saved_actions_and_values: List[Tensor] = []
        self.rewards: list = []

    def select_action(
        self: "ReinforceWithBaselineLearnedByACritic",
        state: List[int],
        possible_actions: List[Tuple[int, int]],
        num_columns: int,
    ) -> Union[Tuple[int, int], bool]:
        self._update_epsilon()
        state_tensor = tensor(state, dtype=float32).unsqueeze(0)
        # Obtient les probabilités d'actions et la valeur d'état
        action_probs, state_value = self.model(state_tensor)
        action_probs = action_probs.squeeze()

        # Déterminer si continue de jouer
        keep_playing = random() < self.keep_playing_threshold

        if random() < self.epsilon:
            return  possible_actions[choice([i for i in range(len(possible_actions))])], keep_playing 

        indices = [self._action_to_index(action, num_columns) for action in possible_actions]
        valid_probs = action_probs[indices]
        valid_probs = softmax(valid_probs, dim=0)
        m = Categorical(valid_probs)
        action_index = m.sample()

        # Enregistrer le log_prob et la valeur d'état pour l'action sélectionnée
        self.saved_actions_and_values.append((m.log_prob(action_index), state_value.squeeze()))

        return possible_actions[action_index.item()], keep_playing

    def update(self: "ReinforceWithBaselineLearnedByACritic") -> None:
        R = 0
        policy_losses = []
        value_losses = []
        returns = []

        # Calcul des retours avec discount pour chaque étape
        for r in self.rewards[::-1]:
            R = r + 0.99 * R
            returns.insert(0, R)

        returns = tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        for (log_prob, value), R in zip(self.saved_actions_and_values, returns):
            # Calcul de la policy loss
            policy_losses.append(-log_prob * (R - value.item()))

            # Calcul de la value loss, utilisant généralement la fonction smooth_l1_loss
            value_losses.append(smooth_l1_loss(value, tensor([R])))

        self.optimizer.zero_grad()

        # Somme des composantes de la perte
        total_loss = stack(policy_losses).sum() + stack(value_losses).sum()
        total_loss.backward()
        self.optimizer.step()

        self.rewards = []
        self.saved_actions_and_values = []
