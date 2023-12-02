from typing import List, Optional

from pydantic import BaseModel

from model.env.cant_stop.box import Box
from model.env.cant_stop.player import Player


class Way(BaseModel):
    boxes: List[Box]
    id: int
    is_won: bool = False
    won_by: Optional[Player] = None

    def way_has_been_won(self: 'Way', by: Player) -> None:
        self.is_won: bool = True
        self.won_by: Player = by
        by.way_won_count += 1
        by.ways_won_id.append(self.id)

        for box in self.boxes:
            box.way_won: bool = True
            box.won_by: Player = by

            if box.is_occupied:
                box.is_occupied: bool = False
                box.who_occupies = None

                if box.occupant_type is not None:
                    box.occupant_type.is_placed: bool = False
                    box.occupant_type.where_is_placed = None
