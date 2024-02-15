
from typing import Dict, List, Optional, Tuple, Union

from pydantic import BaseModel

from src.agent.cant_stop import CantStop as Agent
from src.entity.cant_stop.bonze import Bonze
from src.entity.cant_stop.color import Color
from src.entity.cant_stop.token import Token


class Player(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    agent: Optional[Agent] = None
    color: Color
    bonzes: List[Bonze] = [Bonze(id=1), Bonze(id=2), Bonze(id=3)]
    id: int
    name: str
    playable_bonzes: Dict[int, List[Bonze]] = {}  # key => way_id
    reward: float = 0.0
    tokens: List[Token] = [Token(id=1), Token(id=2), Token(id=3), Token(id=4), Token(id=5), Token(id=6), Token(id=7), Token(id=8), Token(id=9)]
    way_won_count: int = 0
    ways_won_id: List[int] = []

    def get_bonze(self: "Player", way_id: int) -> Union[Bonze, bool]:
        if (playable_bonzes := [bonze for bonze in self.playable_bonzes[way_id]]) != []:
            return playable_bonzes[0]

        return False

    def set_playable_bonzes(self: "Player", way_id: int) -> None:
        self.playable_bonzes[way_id] = []

        for bonze in self.bonzes:
            if bonze.is_placed and bonze.where_is_placed[0] == way_id:
                self.playable_bonzes[way_id].append(bonze)
                return

        self.playable_bonzes[way_id] = [bonze for bonze in self.bonzes if not bonze.is_placed]
        self.agent.playable_bonzes = self.playable_bonzes

    def get_available_possibilities(
        self: "Player",
        possibilities: List[Tuple[int, int]],
        won_ways: List[int],
    ) -> List[Tuple[int, int]]:
        available_possibility: List[Tuple[int, int]] = []

        for possibility in possibilities:
            for way_id in possibility:
                if way_id in won_ways:
                    self.playable_bonzes[way_id]: list = []
                    continue

                self.set_playable_bonzes(way_id)

            available_possibility.append([p for p in possibility if p in self.playable_bonzes and self.playable_bonzes[p]])

        return [sublist for sublist in available_possibility if sublist]

    def reset(self: "Player") -> None:
        self.bonzes: List[Bonze] = [Bonze(id=1), Bonze(id=2), Bonze(id=3)]
        self.playable_bonzes: Dict[int, List[Bonze]] = {}
        self.reward: float = 0.0
        self.tokens: List[Token] = [Token(id=1), Token(id=2), Token(id=3), Token(id=4), Token(id=5), Token(id=6), Token(id=7), Token(id=8), Token(id=9)]
        self.way_won_count: int = 0
        self.ways_won_id: List[int] = []
