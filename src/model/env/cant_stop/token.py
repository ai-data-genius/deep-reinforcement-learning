from typing import Optional, Tuple

from pydantic import BaseModel


class Token(BaseModel):
    is_placed: bool = False
    id: int
    where_is_placed: Optional[Tuple[int]] = None  # way.id, box.id

    def get_icon(self: 'Token') -> str:
        return "●"
