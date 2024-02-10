from itertools import combinations
from typing import List, Tuple

from pydantic import BaseModel

from src.entity.cant_stop.dice import Dice


class DiceRoll(BaseModel):
    allowed_to_play: bool = False  # Si le lancé de dés permet de jouer au moins 1 voie
    by: int  # player_id
    dices: List[Dice]

    def roll_dices(self: 'DiceRoll') -> 'DiceRoll':
        for dice in self.dices:
            dice.roll()

        return self

    def get_possibilities(self: 'DiceRoll') -> List[Tuple[int, int]]:
        n = len(self.dices)
        sum_pairs = []

        for i, j in combinations(range(n), 2):
            for k, l in combinations([x for x in range(n) if x not in (i, j)], 2):
                if (
                    pair := tuple(
                        sorted(
                            (
                                self.dices[i].value + self.dices[j].value,
                                self.dices[k].value + self.dices[l].value
                            )
                        )
                    )
                ) not in sum_pairs:
                    sum_pairs.append(pair)

        return sum_pairs
