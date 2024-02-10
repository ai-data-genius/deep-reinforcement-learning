from typing import List, Optional, Tuple

from pydantic import BaseModel

from src.entity.cant_stop.dice import Dice
from src.entity.cant_stop.dice_roll import DiceRoll


class Turn(BaseModel):
    dices_roll: Optional[DiceRoll] = None
    ended: bool = False
    id: int

    def roll_dices(self: 'Turn', player_id: int, dices: List[Dice]) -> DiceRoll:
        self.dices_roll = DiceRoll(by=player_id, dices=dices).roll_dices()

        return self.dices_roll

    def get_roll_possibilities(self: 'Turn', player_id: int) -> List[int]:
        possibilities: List[int] = []

        while len(possibilities) == 0:
            dices_roll: 'DiceRoll' = self.roll_dices(player_id, [Dice(id=1), Dice(id=2), Dice(id=3), Dice(id=4)])
            possibilities: List[Tuple[int, int]] = dices_roll.get_possibilities()

        if possibilities:
            dices_roll.allowed_to_play: bool = True

        return possibilities
