from typing import List, Optional, Union

from pydantic import BaseModel

from src.entity.cant_stop.bonze import Bonze
from src.entity.cant_stop.player import Player
from src.entity.cant_stop.token import Token


class Box(BaseModel):
    id: int
    has_been_occupied_by: List[int] = []
    is_occupied: bool = False
    occupant_type: Optional[Union[Bonze, Token]] = None
    way_won: bool = False
    who_occupies: Optional[Player] = None
    won_by: Optional[Player] = None
