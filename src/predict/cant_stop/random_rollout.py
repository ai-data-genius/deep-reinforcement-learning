from datetime import datetime
from json import dumps
from os.path import join
from time import time

from src.agent.cant_stop.random_rollout import RandomRollout as Agent
from src.agent.cant_stop.random import Random as RandomAgent
from src.config import folder_paths, nb_columns, nb_episodes
from src.entity.cant_stop.color import Color
from src.entity.cant_stop.player import Player
from src.env.cant_stop import CantStop
from src.metric import Metric


trained_agent: str = "random_rollout"

game: CantStop = CantStop(
    nb_ways=nb_columns,
    players=[
        Player(
            agent=Agent(),
            color=Color(name="red"),
            id=1,
            name="RandomRollout",
        ),
        Player(
            agent=RandomAgent(),
            color=Color(name="green"),
            id=2,
            name="Random",
        ),
    ],
)


def predict() -> None:
    start_time: float = time()

    for _ in range(nb_episodes):
        game.play()
        game.reset()

    with open(
        join(
            folder_paths["metrics"]["cant_stop"],
            f"{trained_agent}--{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}--ep_{nb_episodes}.json",
        ),
        "a",
    ) as file:
        file.write(dumps(Metric(start_time, time(), nb_episodes, game.stats).get()))


if __name__ == "__main__":
    predict()
