
from typing import List, Literal, Tuple, Union

from src.agent.cant_stop import CantStop


class Human(CantStop):
    def __init__(self) -> None:
        super().__init__()

    def keep_playing(self: "Human") -> bool:
        print("Souhaitez-vous continuer l'ascension ? ('o' ou 'n')")
        keep_playing_choice: Literal["o", "n"] = str(input()).strip()

        while keep_playing_choice not in ["o", "n"]:
            print("On a dit 'o' ou 'n' pas :", keep_playing_choice)
            keep_playing_choice: Literal["o", "n"] = str(input()).strip()

        return keep_playing_choice == "o"

    def get_ways_id_with_playable_bonzes(
        self: "Human",
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
        self: "Human",
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

    def select_action(
        self: "Human",
        state: List[int],
        possible_actions: List[Tuple[int, int]],
        num_columns: int,
    ) -> Union[Tuple[int, int], bool]:
        if len(possible_actions) == 1:
            print("Les seuls voies praticables sont :", possible_actions)

            return possible_actions[0], self.keep_playing()

        print("Liste des voies praticables :", possible_actions)
        print(f"Sur quels voies souhaitez-vous envoyer vos grimpeur en ascenssion (entre 0 et {len(possible_actions) - 1})")
        chosen_index: str = input()
        chosen_index: int = int(chosen_index) if chosen_index.isdigit() else -1

        while chosen_index >= len(possible_actions) or chosen_index < 0:
            print(f"Seuls les voies entre 0 et {len(possible_actions) - 1} sont praticables !")
            chosen_index: str = input()
            chosen_index: int = int(chosen_index) if chosen_index.isdigit() else -1

        print("Vous avez décider d'envoyer vos grimpeur en ascension sur les voies :", (chosen_possibilities := possible_actions[chosen_index]))

        return self.chose_ways(chosen_possibilities), self.keep_playing()
