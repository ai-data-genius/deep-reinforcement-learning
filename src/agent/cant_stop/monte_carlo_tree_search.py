from copy import deepcopy
from math import log, sqrt
from random import choice, random
from typing import List, Tuple, Union

from src.agent.cant_stop import CantStop as Agent
from src.entity.cant_stop.player import Player
from src.env.cant_stop import CantStop as Env


class MCTSNode:
    def __init__(
        self: "MCTSNode",
        env: Env,
        player: Player,
        possible_actions: List[Tuple[int, int]],
        action=None,
        parent=None,
    ) -> None:
        self.action: Tuple[int, int] = action
        self.children: List[MCTSNode] = []
        self.env: Env = env
        self.parent: MCTSNode = parent
        self.player: Player = player
        self.untried_actions: List[Tuple[int, int]] = possible_actions
        self.visits: int = 0
        self.wins: float = 0.0

    def uct_select_child(self: "MCTSNode") -> "MCTSNode":
        """Sélectionne un enfant du nœud en utilisant la formule UCT."""

        # Constante d'exploration
        C = 1.414

        # S'assure que les visites sont toujours positives et non nulles pour le calcul
        best_score = float('-inf')
        best_child = None

        for child in self.children:
            # Calcul sécurisé pour éviter division par zéro et log de nombre négatif
            if self.visits == 0 or child.visits == 0:
                score = float('inf')  # Encourage l'exploration de nouveaux nœuds
            else:
                score = child.wins / child.visits + C * sqrt(log(self.visits) / child.visits)

            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def add_child(
        self: "MCTSNode",
        env: Env,
        player: Player,
    ) -> "MCTSNode":
        """Ajoute un nouvel enfant au nœud pour une action donnée."""

        action: Tuple[int, int] = self.untried_actions.pop()
        deepcopy(env)
        env.step(player, action)
        child: MCTSNode = MCTSNode(env, player, self.env.get_possible_actions(self.player), action, self)
        self.children.append(child)

        return child

    def update(self: "MCTSNode", result: float) -> None:
        """Met à jour ce nœud avec le résultat d'une simulation."""

        self.visits += 1
        self.wins += result


class MCTS(Agent):
    def __init__(self, simulation_depth: int = 1, **kwargs) -> None:
        super().__init__(**kwargs)

        self.is_off_policy: bool = True
        self.simulation_depth: int = simulation_depth
        self.root: MCTSNode = None

    def simulate(self: "MCTS", env: Env, player: Player) -> float:
        while not env.is_game_over():
            if (possible_moves := env.get_possible_actions(player)) == []:
                return 0.0

            env.step(player, choice(possible_moves))

        return 1.0 if env.won_by.id == player.id else 0.0

    def select_action(
        self: "MCTS",
        env: Env,
        player: Player,
        possible_actions: List[Tuple[int, int]],
    ) -> Union[Tuple[int, int], bool]:
        self.root: MCTSNode = MCTSNode(env, player, possible_actions)
        untried_actions = deepcopy(self.root.untried_actions)

        for _ in range(self.simulation_depth):
            node: MCTSNode = self.root
            env: Env = deepcopy(env)

            # Phase de sélection
            # nœud est entièrement développé et non terminal
            while node.untried_actions == [] and node.children != []:
                node: MCTSNode = node.uct_select_child()
                env.step(player, node.action)

            # Phase d'expansion
            if node.untried_actions != []:  # si on peut ajouter un enfant
                # ajoute un nouveau nœud enfant
                node: MCTSNode = node.add_child(env, player)
                env.step(player, node.action)

            # Phase de simulation
            result: float = self.simulate(env, player)

            # Phase de rétropropagation
            # remonte dans l'arbre en mettant à jour les nœuds
            while node is not None:
                node.update(result)
                node = node.parent

        return (
            (
                max(self.root.children, key=lambda child: child.wins / child.visits if child.visits else 0).action
                if self.root.children
                else choice(untried_actions)
            ),
            random() >= self.keep_playing_threshold
        )

    def update(self: "MCTS", *args) -> None:
        pass
