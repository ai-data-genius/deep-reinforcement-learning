from random import random
from time import time

from typing import Dict, List, Optional, Tuple, Union

from src.env import Env
from src.entity.cant_stop.board import Board
from src.entity.cant_stop.bonze import Bonze
from src.entity.cant_stop.dice import Dice
from src.entity.cant_stop.player import Player
from src.entity.cant_stop.token import Token
from src.entity.cant_stop.turn import Turn
from src.entity.cant_stop.way import Way


class CantStop(Env):
    def __init__(self: "CantStop", nb_ways: int, players: List[Player]):
        if nb_ways % 2 == 0:
            raise ValueError("Le nombre de voies doit être impair !")

        self.nb_ways: int = nb_ways
        self.players: List[Player] = players
        self.stats: dict = {
            "losses": {player.name: [] for player in self.players},
            "nb_step": 0,
            "step_times": [],
            "reward": {player.name: [] for player in self.players},
            "wins": {player.name: 0 for player in self.players},
        }

        self.reset()

    def determine_play_order(self: "CantStop") -> List[Player]:
        # Lance les dés pour chaque joueur
        dice_rolls = [
            Turn(id=0).roll_dices(player.id, [Dice(id=1), Dice(id=2)])
            for player in self.players
        ]

        # Calcul des scores pour chaque joueur
        player_scores: Dict[int, int] = {
            player.id: sum(dice.value for dice in roll.dices)
            for roll in dice_rolls
            for player in self.players
            if roll.by == player.id
        }

        # Trie les joueurs en fonction de leurs scores
        sorted_player_ids: List[id] = sorted(
            player_scores,
            key=player_scores.get,
            reverse=True,
        )

        # Convertis les IDs triés en objets Player
        self.play_order: List[Player] = [
            next(player for player in self.players if player.id == player_id)
            for player_id in sorted_player_ids
        ]

        return self.play_order

    def get_state(self: "CantStop") -> List[int]:
        state: List[int] = []

        for way in self.board.ways:
            occupied_boxes_by: Dict[int, int] = {
                box.who_occupies.id: box.id
                for box in way.boxes
                if box.is_occupied
            }

            way_state = [way.id, int(way.is_won), way.won_by.id if way.is_won else 0]

            for player in self.players:
                way_state.extend(
                    [
                        player.id,
                        len(way.boxes) - occupied_boxes_by.get(player.id, -1) + 1,
                    ]
                )

            state.extend(way_state)

        state.extend(
            [0] * 4
            if len(self.turns) == 0
            else [dice.value for dice in self.turns[-1].dices_roll.dices]
        )

        return state

    def is_game_over(self: "CantStop") -> Tuple[Optional[Player], bool]:
        for player in self.players:
            if self.nb_way_to_win <= player.way_won_count:
                self.won_by: Player = player
                self.is_over: bool = True
                self.stats["wins"][player.name] += 1

                return self.won_by, self.is_over

    def reset(self: "CantStop") -> None:
        # Adjust the median since x starts from 2
        self.board: Board = Board(median_way_id=(self.nb_ways + 1) // 2 + 1, nb_ways=self.nb_ways)
        self.board.ways: List[Way] = self.board.generate_ways()
        self.is_over: bool = False
        self.nb_way_to_win: int = 3
        self.turns: List[Turn] = []
        self.won_by: Optional[Player] = None

        for player in self.players:
            player.reset()

    def step(
        self: "CantStop",
        player: Player,
        action: Tuple[int, int],
    ) -> float:
        """Return reward."""

        reward = 0.0

        for possibility in action:
            # On retire 2 comme on commence à 2 et que l'index commence à 0.
            way: Way = self.board.ways[possibility - 2]
            player.set_playable_bonzes(way.id)
            chosen_bonze: Union[Bonze, bool] = player.get_bonze(way.id)

            if chosen_bonze is False:
                return reward

            for i, box in enumerate(sorted(way.boxes, key=lambda box: box.id)):
                if not box.is_occupied and player.id not in box.has_been_occupied_by:
                    box.who_occupies: Player = player
                    box.occupant_type: Bonze = chosen_bonze
                    box.is_occupied: bool = True
                    chosen_bonze.is_placed: bool = True
                    chosen_bonze.where_is_placed: Tuple[int, int] = way.id, box.id

                    if i != 0:
                        if not isinstance(way.boxes[i-1].occupant_type, Token):
                            way.boxes[i-1].who_occupies = None
                            way.boxes[i-1].occupant_type = None
                            way.boxes[i-1].is_occupied: bool = False
                            way.boxes[i-1].has_been_occupied_by.append(player.id)

                    reward += 1.0
                    break
                elif (
                    player.id not in box.has_been_occupied_by
                    and box.is_occupied
                    and box.who_occupies != player
                    and i + 1 < len(way.boxes)
                ):
                    # On entre dans cette condition pour appliquer une variante du jeu cant stop
                    # qui consiste à sauter une case si elle est déjà occupée.

                    way.boxes[i+1].who_occupies: Player = player
                    way.boxes[i+1].occupant_type: Bonze = chosen_bonze
                    way.boxes[i+1].is_occupied: bool = True
                    chosen_bonze.is_placed: bool = True
                    chosen_bonze.where_is_placed: Tuple[int, int] = way.id, way.boxes[i+1].id
                    way.boxes[i].has_been_occupied_by.append(player.id)

                    if i != 0:
                        way.boxes[i-1].who_occupies = None
                        way.boxes[i-1].occupant_type = None
                        way.boxes[i-1].is_occupied: bool = False
                        way.boxes[i-1].has_been_occupied_by.append(player.id)

                    reward += 1.5
                    break

        return reward

    def end_of_turn_process(
        self: "CantStop",
        player: Player,
        climbers_have_fallen: bool,
    ) -> None:
        ways_won: List[int] = []

        if not climbers_have_fallen:
            # On vérifie si le joueur à gagner une voie.
            ways_won: List[int] = self.board.player_has_won_ways(player)

        # On rend les grimpeurs au joueur.
        for bonze in player.bonzes:
            bonze.reset()

        # On rend les camps sur une voie gagné au joueur.
        for _player in self.players:
            for token in _player.tokens:
                token.reset(ways_won)

        # On initialise une dictionnaire contenant tous les camps pour chaque voie.
        token_way_box_ids: Dict[int, List[int]] = {way.id: [] for way in self.board.ways}

        # On récupère les emplacements des camps du joueur.
        for token in player.tokens:
            if token.is_placed:
                token_way_box_ids[token.where_is_placed[0]].append(token.where_is_placed[1])

        # On récupère pour chaque voie le camp le plus loin et on enlève les autres.
        token_way_box_id: Dict[int, List[int]] = token_way_box_ids.copy()
        for way_id, boxes_id in token_way_box_id.items():
            if boxes_id:
                for boxe_id in boxes_id:
                    if boxe_id != max(boxes_id):
                        if (
                            token := next(
                                (
                                    token
                                    for token in player.tokens
                                    if token.where_is_placed == (way_id, boxe_id)
                                ),
                                None,
                            )
                        ):
                            token.is_placed: bool = False
                            token.where_is_placed = None

            token_way_box_id[way_id]: int = max(boxes_id) if boxes_id != [] else None

        if climbers_have_fallen:
            for way in self.board.ways:
                # si pas de camp sur la voie on retire tous les mousquetons parcourues sur la voie
                if token_way_box_id[way.id] is None:
                    for box in way.boxes:
                        box.has_been_occupied_by: List[int] = [
                            _id
                            for _id in box.has_been_occupied_by
                            if _id != player.id
                        ]

                        if box.is_occupied and box.who_occupies != player:
                            continue

                        box.is_occupied: bool = False
                        box.occupant_type = None
                        box.who_occupies = None

                    continue

                # si on a un camp sur la voie on retire tous mousquetons parcourues jusqu'au grimpeur
                for i, box in enumerate(sorted(way.boxes, key=lambda box: box.id)):
                    if i < token_way_box_id[way.id]:
                        continue
                    elif i == token_way_box_id[way.id]:
                        box.is_occupied: bool = True
                        box.occupant_type: Token = [
                            token
                            for token in player.tokens
                            if (
                                token.where_is_placed is not None
                                and token.where_is_placed[0] == way.id
                                and token.where_is_placed[1] == box.id
                            )
                        ][0]
                        box.who_occupies: Player = player

                        continue

                    if player.id in box.has_been_occupied_by or player == box.who_occupies:
                        box.has_been_occupied_by: List[int] = [
                            _id
                            for _id in box.has_been_occupied_by
                            if _id != player.id
                        ]

                        if box.is_occupied and box.who_occupies != player:
                            continue

                        box.is_occupied: bool = False
                        box.occupant_type = None
                        box.who_occupies = None
        else:
            token_available: List[Token] = [token for token in player.tokens if not token.is_placed]

            for way in self.board.ways:
                for i, box in enumerate(sorted(way.boxes, key=lambda box: box.id)):
                    if box.is_occupied and box.who_occupies == player and isinstance(box.occupant_type, Bonze):
                        if token_way_box_id[way.id] is not None:
                            # On déplace le camp déjà présent sur la voie.
                            token = next(
                                (
                                    token
                                    for token in player.tokens
                                    if token.where_is_placed == (way.id, token_way_box_id[way.id])
                                ),
                                None,
                            )

                            box.occupant_type: Token = token
                            token.where_is_placed: Tuple[int, int] = way.id, box.id
                            # On rajoute la position du token qu'on vient d'utiliser.
                            token_way_box_ids[way.id].append(box.id)
                        elif len(token_available) != 0:
                            # On place un nouveau camp sur la voie.
                            box.occupant_type: Token = token_available[0]
                            token_available[0].is_placed: bool = True
                            token_available[0].where_is_placed: Tuple[int, int] = way.id, box.id
                            token_available.pop(0)  # On retire le camp utilisé.
                            # On rajoute la position du token qu'on vient d'utiliser.
                            token_way_box_ids[way.id].append(box.id)
                        else:
                            # Si plus de camps et pas de camp présent sur la voie,
                            # on retire le grimpeur du mousqueton.
                            box.occupant_type = None
                            box.is_occupied: bool = False
                            box.who_occupies = None

                            # On retire la présence du joueur sur tous les mousquetons de la voie.
                            for j in range(i):
                                way.boxes[j].has_been_occupied_by: List[int] = [
                                    _id
                                    for _id in way.boxes[j].has_been_occupied_by
                                    if _id != player.id
                                ]

            # On parcourt toutes les voies et tous les mousquetons pour y retirer les anciens camps. 
            for way in self.board.ways:
                token_place_to_remove = None

                for i, box in enumerate(sorted(way.boxes, key=lambda box: box.id)):
                    if (
                        box.is_occupied
                        and box.who_occupies == player
                        and isinstance(box.occupant_type, Token)
                        and sorted(token_way_box_ids[way.id])[-1] != box.id
                    ):
                        # On enlève l'ancien camp de la voie.
                        box.is_occupied: bool = False
                        box.occupant_type = None
                        box.who_occupies = None
                        box.has_been_occupied_by.append(player.id)
                        token_place_to_remove = (way.id, box.id)
                        break

                if token_place_to_remove is None:
                    continue

                for token in player.tokens:
                    if (
                        token.where_is_placed is not None
                        and token.where_is_placed[0] == token_place_to_remove[0]
                        and token.where_is_placed[1] == token_place_to_remove[1]
                    ):
                        # On remet le camp dans les mains du joueur.
                        token.is_placed: bool = False
                        token.where_is_placed = None
                        break

    def get_possible_actions(self: "CantStop", player: Player) -> List[Tuple[int, int]]:
        self.turns.append((turn := Turn(id=len(self.turns) + 1)))
        turn.roll_dices(player.id, [Dice(id=1), Dice(id=2), Dice(id=3), Dice(id=4)])

        # si on a qu'une valeur on la double pour en obtenir deux
        return [
            action
            if len(action) == 2
            else action + action
            for action in player.get_available_possibilities(
                turn.get_roll_possibilities(player.id),
                [way.id for way in self.board.ways if way.is_won],
            )
        ]

    def _update(
        self: "CantStop",
        player: Player,
        current_state: List[int],
        action: Tuple[int, int],
        reward: float,
    ) -> None:
        player.agent.remember(
            current_state,
            action,
            reward,
            self.get_state(),  # <=> next_state
            self.is_over,
        )

        player.agent.update(
            current_state,
            action,
            reward,
            self.get_state(),  # <=> next_state
            self.is_over,
        )

    def play(
        self: "CantStop",
        render: bool = False,
    ) -> None:
        self.determine_play_order() 

        while not self.is_over:
            for player in self.players:
                if self.is_over:
                    break

                keep_playing: bool = True
                climbers_have_fallen: bool = False

                while keep_playing:
                    reward: float = 0.0
                    possible_actions: List[Tuple[int, int]] = self.get_possible_actions(player)
                    current_state: List[int] = self.get_state()

                    if possible_actions == []:
                        climbers_have_fallen: bool = True
                        action = [1, 1]
                        keep_playing = False

                        if render:
                            print("Miséricorde ! Les grimpeurs viennent de tomber... Retour aux camps obligé !")

                        continue

                    start_time: float = time()

                    if not player.agent.is_off_policy and not player.agent.is_rollout:
                        action, keep_playing = player.agent.select_action(
                            current_state,
                            possible_actions,
                            self.nb_ways,
                        )
                    elif player.agent.is_rollout:
                        best_action = None
                        best_reward = float('-inf')

                        for _action in possible_actions:
                            action_reward = player.agent.rollout(self, player, _action)

                            if action_reward > best_reward:
                                best_reward = action_reward
                                best_action = _action

                        action = best_action
                        keep_playing = random() >= player.agent.keep_playing_threshold
                    else:
                        action, keep_playing = player.agent.select_action(self, player, possible_actions)

                    reward += self.step(player, action)
                    player.reward += reward

                    if player.agent.is_mc_policy:
                        player.agent.rewards.append(reward)

                    self.stats["step_times"].append(time() - start_time)
                    self.stats["nb_step"] += 1

                    if render:
                        self.board.display()

                self.end_of_turn_process(player, climbers_have_fallen)

                if render:
                    self.board.display()

                self.is_game_over()

                if not player.agent.update_each_episode:
                    self._update(player, current_state, action, reward)

        for player in self.players:
            player.reward = player.reward + 5 if self.won_by == player else -1.0
            self.stats["reward"][player.name].append(player.reward)

            if not player.agent.is_off_policy:
                self.stats["losses"][player.name] = player.agent.cumulative_losses

            if player.agent.update_each_episode:
                player.agent.update()
            else:
                self._update(
                    player,
                    self.get_state(),
                    [1, 1],
                    player.reward,
                )

        if render:
            print(f"Félicitations {self.won_by.name} ! 🎉 Vous avez remporté la partie. 🥳")
