from time import time

from src.agent.cant_stop.monte_carlo_tree_search import MCTS as Agent
from src.agent.cant_stop.random import Random as RandomAgent
from src.config import nb_columns, nb_episodes
from src.entity.cant_stop.color import Color
from src.entity.cant_stop.player import Player
from src.env.cant_stop import CantStop


game: CantStop = CantStop(
    nb_ways=nb_columns,
    players=[
        Player(
            agent=Agent(num_columns=nb_columns),
            color=Color(name="red"),
            id=1,
            name="MCTS",
        ),
        Player(
            agent=RandomAgent(),
            color=Color(name="green"),
            id=2,
            name="Random",
        ),
    ],
)


def train() -> None:
    win_stat: dict = {player.id: 0 for player in game.players}
    start_time=time()

    for _ in range(nb_episodes):
        game.play()
        win_stat[game.won_by.id] += 1
        game.reset()

    end_time=time()

    print(f"Le temps d'exécution de la boucle est de {end_time - start_time} secondes.")
    print(f"Stats: {win_stat}")


if __name__ == "__main__":
    train()
