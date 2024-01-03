from typing import Dict, List, Literal, Tuple, Union

from pydantic import BaseModel

from model.env.cant_stop.bonze import Bonze
from model.env.cant_stop.color import Color
from model.env.cant_stop.token import Token
from model.env.cant_stop.turn import Turn


class Player(BaseModel):
    color: Color
    bonzes: List[Bonze] = [Bonze(id=1), Bonze(id=2), Bonze(id=3)]
    id: int
    name: str
    playable_bonzes: Dict[int, List[Bonze]] = {}  # key => way_id
    tokens: List[Token] = [Token(id=1), Token(id=2), Token(id=3), Token(id=4), Token(id=5), Token(id=6), Token(id=7), Token(id=8), Token(id=9)]
    way_won_count: int = 0
    ways_won_id: List[int] = []

    def get_bonze(self: 'Player', way_id: int) -> Union[Bonze, bool]:
        if (playable_bonzes := [bonze for bonze in self.playable_bonzes[way_id]]) != []:
            return playable_bonzes[0]

        return False

    def chose_possibilities(
        self: 'Player',
        turn: Turn,
        won_ways: List[int],
    ) -> Union[bool, List[int]]:
        possibilities: List[int] = self.get_available_possibilities(turn.get_roll_possibilities(self.id), won_ways)

        if possibilities == []:
            return False

        if len(possibilities) == 1:
            print("Les seuls voies praticables sont :", possibilities)
            chosen_index: int = 0

            return possibilities[chosen_index]

        print("Liste des voies praticables :", possibilities)
        print(f"Sur quels voies souhaitez-vous envoyer vos grimpeur en ascenssion (entre 0 et {len(possibilities) - 1})")
        chosen_index: int = int(input())

        while chosen_index >= len(possibilities) or chosen_index < 0:
            print(f"Seuls les voies entre 0 et {len(possibilities) - 1} sont praticables !")
            chosen_index: int = int(input())

        print("Vous avez décider d'envoyer vos grimpeur en ascension sur les voies :", (chosen_possibilities := possibilities[chosen_index]))

        return chosen_possibilities

    def set_playable_bonzes(self: 'Player', way_id: int) -> None:
        self.playable_bonzes[way_id] = []

        for bonze in self.bonzes:
            if bonze.is_placed and bonze.where_is_placed[0] == way_id:
                self.playable_bonzes[way_id].append(bonze)
                return

        self.playable_bonzes[way_id] = [bonze for bonze in self.bonzes if not bonze.is_placed]

    def get_available_possibilities(
        self: 'Player',
        possibilities: List[Tuple[int]],
        won_ways: List[int],
    ) -> List[Tuple[int]]:
        available_possibility: List[Tuple[int]] = []

        for possibility in possibilities:
            for way_id in possibility:
                if way_id in won_ways:
                    self.playable_bonzes[way_id]: list = []
                    continue

                self.set_playable_bonzes(way_id)

            available_possibility.append([p for p in possibility if p in self.playable_bonzes and self.playable_bonzes[p]])

        return [sublist for sublist in available_possibility if sublist]

    def get_ways_id_with_playable_bonzes(
        self: 'Player',
        chosen_possibilities: List[int],
    ) -> Tuple[int, List[int]]:
        nb_playable_bonzes_for_chosen_possibilities: int = 0
        ways_id_with_playable_bonzes: List[int] = []
        chosen_possibilities_bonzes_id: List[int] = []

        for way_id in chosen_possibilities:
            for bonze in self.playable_bonzes[way_id]:
                if bonze.id not in chosen_possibilities_bonzes_id:
                    chosen_possibilities_bonzes_id.append(bonze.id)
                    nb_playable_bonzes_for_chosen_possibilities += 1

            if len(self.playable_bonzes[way_id]) > 0:
                ways_id_with_playable_bonzes.append(way_id)

        return nb_playable_bonzes_for_chosen_possibilities, ways_id_with_playable_bonzes

    def chose_ways(
        self: 'Player',
        chosen_possibilities: List[int]
    ):
        (
            nb_playable_bonzes_for_chosen_possibilities,
            ways_id_with_playable_bonzes
        ) = self.get_ways_id_with_playable_bonzes(chosen_possibilities)

        if nb_playable_bonzes_for_chosen_possibilities != 1:
            return chosen_possibilities

        if len(ways_id_with_playable_bonzes) == 1:
            return ways_id_with_playable_bonzes

        if (
            len(ways_id_with_playable_bonzes) == 2
            and ways_id_with_playable_bonzes[0] == ways_id_with_playable_bonzes[1]
        ):
            return ways_id_with_playable_bonzes

        print(f"Vous avez besoin de choisir entre ces voies {ways_id_with_playable_bonzes}")
        chosen_possibilities: int = int(input())

        while chosen_possibilities not in ways_id_with_playable_bonzes:
            print(f"Les voies praticables que vous pouvez choisir sont : {ways_id_with_playable_bonzes} !")
            chosen_possibilities: int = int(input())

        return [chosen_possibilities]

    def keep_playing(self: 'Player') -> bool:
        print("Souhaitez-vous continuer l'ascension ? ('o' ou 'n')")
        keep_playing_choice: Literal["o", "n"] = str(input()).strip()

        while keep_playing_choice not in ["o", "n"]:
            print("On a dit 'o' ou 'n' pas :", keep_playing_choice)
            keep_playing_choice: Literal["o", "n"] = str(input()).strip()

        return keep_playing_choice == "o"
