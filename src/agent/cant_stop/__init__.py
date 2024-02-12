from typing import Callable, Dict, List, Optional, Tuple

from torch import load
from torch.optim import Optimizer

from src.agent import Agent
from src.agent.tool.replay_buffer import ReplayBuffer
from src.entity.cant_stop.bonze import Bonze
from src.network import Network


class CantStop(Agent):
    def __init__(
        self: "CantStop",
        batch_size: Optional[int] = None,
        criterion: Optional[Callable] = None,
        decay_rate: Optional[float] = None,
        epsilon: Optional[float] = None,
        gamma: Optional[float] = None,
        is_policy_gradient: bool = False,
        is_off_policy: bool = False,
        model: Optional[Network] = None,
        memory: Optional[ReplayBuffer] = None,
        memory_size: int = 20_0000,
        num_columns: Optional[int] = None,
        optimizer: Optional[Optimizer] = None,
    ) -> None:
        self.batch_size: Optional[int] = batch_size
        self.criterion: Optional[Callable] = criterion
        self.decay_rate: Optional[float] = decay_rate
        self.epsilon: Optional[float] = epsilon
        self.gamma: Optional[float] = gamma
        self.keep_playing_threshold: float = .5
        self.model: Optional[Network] = model
        self.is_policy_gradient: bool = is_policy_gradient
        self.is_off_policy: bool = is_off_policy
        self.memory: Optional[ReplayBuffer] = memory
        self.memory_size: int = memory_size
        self.num_columns: Optional[int] = num_columns
        self.optimizer: Optional[Optimizer] = optimizer
        self.playable_bonzes: Dict[int, List[Bonze]] = {}

    def _action_to_index(
        self: "CantStop",
        action: Tuple[int, int],
        num_columns: int,
    ) -> int:
        # -2 pour l'ajustement pour le décalage de
        # numérotation (ex: 2 à 12 au lieu de 1 à 11)
        return (
            (action[0] - 2)
            * num_columns
            + (
                (action[1] - 1)
                if len(action) == 2
                else (action[0] - 2)
            )
        )

    def _update_epsilon(self: "CantStop") -> None:
        self.epsilon = max(self.epsilon * self.decay_rate, 0.01)

    def load_model(self: "CantStop", path: str) -> None:
        self.model.load_state_dict(load(path))

    def remember(
        self: "CantStop",
        state: List[int],
        action: Tuple[int, int],
        reward: float,
        next_state: List[int],
        done: bool,
    ) -> None:
        pass

    def update(self: "CantStop", *args, **kwargs) -> None:
        pass
