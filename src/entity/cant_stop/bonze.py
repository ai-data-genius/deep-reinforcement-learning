from typing import Optional, Tuple

from pydantic import BaseModel


class Bonze(BaseModel):
    is_placed: bool = False
    id: int
    where_is_placed: Optional[Tuple[int, int]] = None  # way.id, box.id

    def get_icon(self: 'Bonze') -> str:
        return "▲"

    def reset(self: 'Bonze') -> None:
        """On rend les grimpeurs au joueur à la fin de chaque tour."""

        self.is_placed: bool = False
        self.where_is_placed = None
