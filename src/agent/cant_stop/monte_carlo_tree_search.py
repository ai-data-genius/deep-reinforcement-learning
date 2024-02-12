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
        action=None,
        parent=None,
    ) -> None:
        self.action: Tuple[int, int] = action
        self.children: List[MCTSNode] = []
        self.env: Env = env
        self.parent: MCTSNode = parent
        self.player: Player = player
        self.state: List[int] = env.get_state().copy()
        self.untried_actions: List[Tuple[int, int]] = env.get_possible_actions(player)
        self.visits: int = 0
        self.reward: float = 0.0

    def expand_node(self):
        """Étend un nœud en générant des enfants pour chaque action possible."""

        possible_actions = self.env.get_possible_actions(self.player)

        for action in possible_actions:
            self.env.step(self.player, action)
            self.children.append(MCTSNode(self.env, self.player, action, self))

    def uct_select_child(self: "MCTSNode") -> "MCTSNode":
        """Sélectionne un enfant du nœud en utilisant la formule UCT."""

        if not self.children:
            # Si le nœud n'a pas d'enfants, il doit d'abord être étendu
            self.expand_node()

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
                score = child.reward / child.visits + C * sqrt(log(self.visits) / child.visits)

            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def add_child(
        self: "MCTSNode",
        action: Tuple[int, int],
        env: Env,
        player: Player,
    ) -> "MCTSNode":
        """Ajoute un nouvel enfant au nœud pour une action donnée."""

        child = MCTSNode(env, player, action, self)
        self.untried_actions.remove(action)
        self.children.append(child)

        return child

    def update(self: "MCTSNode", result: float) -> None:
        """Met à jour ce nœud avec le résultat d'une simulation."""

        self.visits += 1
        self.reward += result


class MCTS(Agent):
    def __init__(self, simulation_depth: int = 10, **kwargs) -> None:
        super().__init__(**kwargs)

        self.is_off_policy: bool = True
        self.simulation_depth: int = simulation_depth
        self.root: MCTSNode = None

    def select_action(
        self: "MCTS",
        env: Env,
        player: Player,
    ) -> Union[Tuple[int, int], bool]:
        state: List[int] = env.get_state()
        self.root: MCTSNode = MCTSNode(env, player)

        for _ in range(self.simulation_depth):
            node: MCTSNode = self.root
            state: List[int] = state.copy()

            # Phase de sélection
            # nœud est entièrement développé et non terminal
            while node.untried_actions == [] and node.children != []:
                node: MCTSNode = node.uct_select_child()
                env.step(player, node.action)

            # Phase d'expansion
            if node.untried_actions:  # si on peut ajouter un enfant
                action: List[int, int] = choice(node.untried_actions)
                env.step(player, action)
                # ajoute un nouveau nœud enfant
                node: MCTSNode = node.add_child(action, env, player)

            # Phase de simulation
            while env.get_possible_actions(player) != []:
                if (actions := env.get_possible_actions(player)) == []:
                    break

                env.step(player, choice(actions))

            # Phase de rétropropagation
            # remonte dans l'arbre en mettant à jour les nœuds
            while node is not None:
                if node.action is None:
                    break

                node.update(env.step(node.player, node.action))
                node = node.parent

        # retourne l'action du meilleur enfant et si on doit continuer à jouer
        if (chosen_child := self.root.uct_select_child()) is None:
            while len(self.root.children) == 0:
                self.root.expand_node()

            chosen_child = self.root.children[0]

        return (
            chosen_child.action,
            random() >= self.keep_playing_threshold
        )

    def update(self: "MCTS", *args) -> None:
        pass
