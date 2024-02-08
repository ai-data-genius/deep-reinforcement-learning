from random import randint
from typing import Optional

from pydantic import BaseModel


class Dice(BaseModel):
    face: int = 6
    id: int
    value: Optional[int] = None
    has_been_launched: bool = False

    def roll(self: 'Dice') -> None:
        if self.has_been_launched:
            return

        self.value: int = randint(1, self.face)
        self.has_been_launched: bool = True
