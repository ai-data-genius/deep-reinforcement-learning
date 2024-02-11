from random import choice, random
from typing import List, Tuple, Union

from torch import (
    cat,
    float32,
    softmax,
    Tensor,
    tensor,
)
from torch.distributions import Categorical

from src.agent.cant_stop import CantStop


class Reinforce(CantStop):
    def __init__(self: "Reinforce", **kwargs) -> None:
        super().__init__(**kwargs)

        self.saved_log_probs: List[Tensor] = []
        self.rewards: list = []

    def select_action(
        self: "Reinforce",
        state: List[int],
        possible_actions: List[Tuple[int, int]],
        num_columns: int,
    ) -> Union[Tuple[int, int], bool]:
        self._update_epsilon()
        state_tensor = tensor(state, dtype=float32).unsqueeze(0)
        action_probs, keep_playing_prob = self.model(state_tensor)
        action_probs = action_probs.squeeze()

        keep_playing = keep_playing_prob.item() >= self.keep_playing_threshold

        if random() < self.epsilon:
            return choice(possible_actions), keep_playing

        indices = [self._action_to_index(action, num_columns) for action in possible_actions]
        valid_probs = action_probs[indices]
        valid_probs = softmax(valid_probs, dim=0)
        m = Categorical(valid_probs)
        action_index = m.sample()

        chosen_action = possible_actions[action_index.item()]
        self.saved_log_probs.append(m.log_prob(action_index))

        return chosen_action, keep_playing

    def update(self: "Reinforce") -> None:
        R: int = 0
        policy_loss: List[Tensor] = []
        returns: List[float] = []

        # Calcul des retours avec discount pour chaque étape
        for r in self.rewards[::-1]:
            R = r + 0.99 * R
            returns.insert(0, R)

        # Conversion en Tensor et normalisation des retours
        returns: Tensor = tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)

        policy_loss = [loss.unsqueeze(0) for loss in policy_loss]

        self.optimizer.zero_grad()
        policy_loss: Tensor = cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()

        self.rewards: list = []
        self.saved_log_probs: list = []
