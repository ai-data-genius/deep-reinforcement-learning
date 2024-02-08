from random import random, choice
from typing import List, Tuple, Union

from src.agent.cant_stop import CantStop


class Random(CantStop):
    def __init__(self) -> None:
        super().__init__()

    def select_action(
        self: "CantStop",
        state: List[int],
        possible_actions: List[Tuple[int, int]],
        num_columns: int,
    ) -> Union[Tuple[int, int], bool]:
        return choice(possible_actions), random() >= self.keep_playing_threshold
