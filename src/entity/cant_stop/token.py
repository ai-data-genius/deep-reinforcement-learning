from typing import List, Optional, Tuple

from pydantic import BaseModel


class Token(BaseModel):
    is_placed: bool = False
    id: int
    where_is_placed: Optional[Tuple[int, int]] = None  # way.id, box.id

    def get_icon(self: 'Token') -> str:
        return "●"

    def reset(self: 'Token', ways_won: List[int]) -> None:
        """Si le camp est placé sur une voie qui est gagné, on le rend au joueur."""

        if self.is_placed and self.where_is_placed[0] in ways_won:
            self.is_placed: bool = False
            self.where_is_placed = None
