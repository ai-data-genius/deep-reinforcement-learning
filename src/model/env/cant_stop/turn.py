from typing import List, Tuple

from pydantic import BaseModel

from model.env.cant_stop.dice import Dice
from model.env.cant_stop.dice_roll import DiceRoll


class Turn(BaseModel):
    dice_rolls: List[DiceRoll] = []
    ended: bool = False
    id: int

    def roll_dice(self: 'Turn', player_id: int, dices: List[Dice]) -> DiceRoll:
        self.dice_rolls.append(
            (
                dice_roll := DiceRoll(
                    by=player_id,
                    dices=dices,
                    id=len(self.dice_rolls) + 1,
                ).roll_dices()
            )
        )

        return dice_roll

    def get_roll_available_possibilities(self: 'Turn', player_id: int) -> List[int]:
        possibilities: List[int] = []

        while len(possibilities) == 0:
            dice_roll: 'DiceRoll' = self.roll_dice(player_id, [Dice(id=1), Dice(id=2), Dice(id=3), Dice(id=4)])
            possibilities: List[Tuple[int]] = dice_roll.get_possibilities()

        dice_roll.allowed_to_play: bool = True

        return possibilities
