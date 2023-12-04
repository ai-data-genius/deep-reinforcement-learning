from typing import Dict, Literal, List, Optional, Tuple, Union

from model.env.cant_stop.board import Board
from model.env.cant_stop.bonze import Bonze
from model.env.cant_stop.dice import Dice
from model.env.cant_stop.player import Player
from model.env.cant_stop.token import Token
from model.env.cant_stop.turn import Turn
from model.env.cant_stop.way import Way


class CantStop:
    def __init__(self: 'CantStop', nb_ways: int, players: List[Player]):
        if nb_ways % 2 == 0:
            raise ValueError("Le nombre de voies doit être impair !")

        self.players: List[Player] = players
        self.board: Board = Board(
            median_way_id=(nb_ways + 1) // 2 + 1,  # Adjust the median since x starts from 2,
            nb_ways=nb_ways,
        )
        self.board.ways: List[Way] = self.board.generate_ways()
        self.turns: List[Turn] = []
        self.nb_way_to_win: int = 3
        self.won_by: Optional[Player] = None
        self.is_over: bool = False

    def create_turn(self: 'CantStop') -> Turn:
        self.turns.append((turn := Turn(id=len(self.turns) + 1)))

        return turn

    def determine_play_order(self: 'CantStop') -> List[Player]:
        turn: Turn = self.create_turn()

        # Lancez les dés pour chaque joueur
        for player in self.players:
            turn.roll_dice(player.id, [Dice(id=1), Dice(id=2)])

        # Calcul des scores pour chaque joueur
        player_scores: Dict[int, int] = {player.id: sum(dice.value for dice in roll.dices) for roll in turn.dice_rolls for player in self.players if roll.by == player.id}

        # Triez les joueurs en fonction de leurs scores
        sorted_player_ids: List[id] = sorted(player_scores, key=player_scores.get, reverse=True)

        # Convertissez les IDs triés en objets Player
        self.play_order: List[Player] = [next(player for player in self.players if player.id == player_id) for player_id in sorted_player_ids]

        turn.ended: bool = True
        self.turns.append(turn)

        return self.play_order

    def is_finished(self: 'CantStop') -> Tuple[Optional[Player], bool]:
        for player in self.players:
            if self.nb_way_to_win == player.way_won_count:
                self.won_by: Player = player
                self.is_over: bool = True

                break

        return self.won_by, self.is_over

    def play_a_turn(self: 'CantStop') -> Turn:
        turn: Turn = self.create_turn()

        for player in self.play_order:
            if self.is_over:
                break

            print(f"\033[1;31mAu tour des grimpeurs de {player.name} de faire l'ascension !\033[m")

            keep_playing: bool = True
            climbers_have_fallen: bool = False

            while keep_playing:
                chosen_possibilities: Union[bool, List[int]] = player.chose_possibilities(turn, [way.id for way in self.board.ways if way.is_won])

                if chosen_possibilities == False:
                    climbers_have_fallen: bool = True
                    print("Miséricorde ! Les grimpeurs viennent de tomber... Retour aux camps obligé !")
                    break

                for possibility in player.chose_ways(chosen_possibilities):
                    way: Way = self.board.ways[possibility - 2]  # on retire 2 comme on commence à 2 et que l'index commence à 0
                    player.set_playable_bonzes(way.id)
                    chosen_bonze: Bonze = player.chose_bonze(way.id)

                    for i, box in enumerate(sorted(way.boxes, key=lambda box: box.id)):
                        if not box.is_occupied and player.id not in box.has_been_occupied_by:
                            box.who_occupies: Player = player
                            box.occupant_type: Bonze = chosen_bonze
                            box.is_occupied: bool = True
                            chosen_bonze.is_placed: bool = True
                            chosen_bonze.where_is_placed: Tuple[int] = way.id, box.id

                            print(f"Votre grimpeur {chosen_bonze.id} a atteint le mousqueton {box.id + 1} de la voie {way.id}")

                            if i != 0:
                                if not isinstance(way.boxes[i-1].occupant_type, Token):
                                    way.boxes[i-1].who_occupies = None
                                    way.boxes[i-1].occupant_type = None
                                    way.boxes[i-1].is_occupied: bool = False
                                    way.boxes[i-1].has_been_occupied_by.append(player.id)

                            break
                        elif (
                            player.id not in box.has_been_occupied_by
                            and box.is_occupied
                            and box.who_occupies != player
                            and i + 1 < len(way.boxes)
                        ):
                            way.boxes[i+1].who_occupies: Player = player
                            way.boxes[i+1].occupant_type: Bonze = chosen_bonze
                            way.boxes[i+1].is_occupied: bool = True
                            chosen_bonze.is_placed: bool = True
                            chosen_bonze.where_is_placed: Tuple[int] = way.id, way.boxes[i+1].id
                            way.boxes[i].has_been_occupied_by.append(player.id)

                            if i != 0:
                                way.boxes[i-1].who_occupies = None
                                way.boxes[i-1].occupant_type = None
                                way.boxes[i-1].is_occupied: bool = False
                                way.boxes[i-1].has_been_occupied_by.append(player.id)

                            print(f"""\
                                Un autre grimpeur est déjà présent sur le mousqueton {box.id + 1}, \
                                votre grimpeur {chosen_bonze.id} a sauté sur le mousqueton \
                                {way.boxes[i+1].id + 1} de la voie {way.id}. \
                            """)

                            break

                self.board.display()

                print("Souhaitez-vous continuer l'ascension ? ('o' ou 'n')")
                keep_playing_choice: Literal["o", "n"] = str(input()).strip()

                while keep_playing_choice not in ["o", "n"]:
                    print("On a dit 'o' ou 'n' pas :", keep_playing_choice)
                    keep_playing_choice: Literal["o", "n"] = str(input()).strip()

                keep_playing: bool = keep_playing_choice == "o"

            for way in self.board.ways:
                if sorted(way.boxes, key=lambda box: box.id)[-1].who_occupies == player:
                    way.way_has_been_won(player)

            for bonze in player.bonzes:
                bonze.is_placed: bool = False
                bonze.where_is_placed = None

            token_way_box_ids: Dict[int, List[int]] = {way.id: [] for way in self.board.ways}

            for token in player.tokens:
                if token.is_placed:
                    token_way_box_ids[token.where_is_placed[0]].append(token.where_is_placed[1])

            token_way_box_id: Dict[int, List[int]] = token_way_box_ids.copy()
            for way_id, boxes_id in token_way_box_id.items():
                token_way_box_id[way_id]: int = sorted(boxes_id)[-1] if boxes_id != [] else None

            if climbers_have_fallen:
                for way in self.board.ways:
                    if token_way_box_id[way.id] is None:  # si pas de camp sur la voie on retire tous les mousquetons parcourues sur la voie
                        for box in way.boxes:
                            box.has_been_occupied_by: List[int] = [_id for _id in box.has_been_occupied_by if _id != player.id]

                            if box.is_occupied and box.who_occupies != player:
                                continue

                            box.is_occupied: bool = False
                            box.occupant_type = None
                            box.who_occupies = None

                        continue

                    for i, box in enumerate(sorted(way.boxes, key=lambda box: box.id)):  # si on a un camp sur la voie on retire tous mousquetons parcourues jusqu'au grimpeur
                        if i < token_way_box_id[way.id]:
                            continue
                        elif i == token_way_box_id[way.id]:
                            box.is_occupied: bool = True
                            box.occupant_type: Token = [token for token in player.tokens if token.where_is_placed is not None and token.where_is_placed[0] == way.id and token.where_is_placed[1] == box.id][0]
                            box.who_occupies: Player = player

                            continue

                        if player.id in box.has_been_occupied_by or player == box.who_occupies:
                            box.has_been_occupied_by: List[int] = [_id for _id in box.has_been_occupied_by if _id != player.id]

                            if box.is_occupied and box.who_occupies != player:
                                continue

                            box.is_occupied: bool = False
                            box.occupant_type = None
                            box.who_occupies = None
            else:
                token_available: List[Token] = [token for token in player.tokens if not token.is_placed]

                if len(token_available) == 0:
                    print("Vous n'avez plus de camp à placer !")

                for way in self.board.ways:
                    for i, box in enumerate(sorted(way.boxes, key=lambda box: box.id)):
                        if box.is_occupied and box.who_occupies == player and isinstance(box.occupant_type, Bonze):
                            if len(token_available) != 0:
                                box.occupant_type: Token = token_available[0]
                                token_available[0].is_placed: bool = True
                                token_available[0].where_is_placed: Tuple[int] = way.id, box.id
                                token_available.pop(0)  # on retire le token utilisé
                                token_way_box_ids[way.id].append(box.id)  # on rajoute la position du token qu'on vient d'utiliser

                                if len(token_available) == 0:
                                    print("Vous venez d'épuiser tous vos camps !")
                            elif token_way_box_ids[way.id] is not None:
                                box.occupant_type = None
                                box.is_occupied: bool = False
                                box.who_occupies = None

                                for j in range(i):
                                    way.boxes[j].has_been_occupied_by: List[int] = [_id for _id in way.boxes[j].has_been_occupied_by if _id != player.id]

                for way in self.board.ways:
                    for i, box in enumerate(sorted(way.boxes, key=lambda box: box.id)):
                        if (
                            box.is_occupied
                            and box.who_occupies == player
                            and isinstance(box.occupant_type, Token)
                            and sorted(token_way_box_ids[way.id])[-1] != box.id
                        ):
                            # on retire le token du board
                            box.is_occupied: bool = False
                            box.occupant_type = None
                            box.who_occupies = None
                            box.has_been_occupied_by.append(player.id)

                            for token in player.tokens:
                                if (
                                    token.where_is_placed is not None
                                    and token.where_is_placed[0] == way.id
                                    and token.where_is_placed[1] == box.id
                                ):  # on remet le token dans les mains du joueur
                                    token.is_placed: bool = False
                                    token.where_is_placed = None
                                    token_available.append(token)  # on rajoute le token au token_available
                                    break

            self.board.display()

    def play(self: 'CantStop') -> None:
        print("Voici l'ordre de passage des équipes pour les ascensions :", [player.name for player in self.determine_play_order()])

        while not self.is_over:
            self.play_a_turn()
            self.is_finished()
