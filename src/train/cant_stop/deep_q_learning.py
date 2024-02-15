from datetime import datetime
from json import dumps
from os.path import join
from time import time

from torch.nn import SmoothL1Loss  # Huber
from torch.optim import Adam

from src.agent.cant_stop.deep_q_learning import DeepQLearning as DQLAgent
from src.agent.cant_stop.random import Random as RandomAgent
from src.agent.tool.replay_buffer import ReplayBuffer
from src.config import folder_paths, nb_columns, nb_episodes
from src.entity.cant_stop.color import Color
from src.entity.cant_stop.player import Player
from src.env.cant_stop import CantStop
from src.metric import Metric
from src.network.deep_q import DeepQNet


trained_agent: str = "deep_q_learning"

agent: DQLAgent = DQLAgent(
    batch_size=10,
    criterion=SmoothL1Loss(),
    decay_rate=.9,
    epsilon=1.0,
    gamma=.99,
    model=(model := DeepQNet(
        # le nombre de colonne * 7 caractéristiques chacune + 4 dès
        input_size=(nb_columns * 7) + 4,
        hidden_size=28,  # à tuner
        output_size=pow(nb_columns, 2) + 1,  # + 1 pour keep_playing action
    )),
    memory=ReplayBuffer(10_000),
    num_columns=nb_columns,
    optimizer=Adam(model.parameters(), lr=.01),
)

game: CantStop = CantStop(
    nb_ways=nb_columns,
    players=[
        Player(
            agent=agent,
            color=Color(name="red"),
            id=1,
            name="DQLAgent",
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
    start_time: float = time()

    for _ in range(nb_episodes):
        game.play()
        game.reset()

    agent.model.save(
        folder_paths["models"]["cant_stop"],
        f"{trained_agent}.pth",
    )

    with open(
        join(
            folder_paths["metrics"]["cant_stop"],
            f"{trained_agent}--{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}--ep_{nb_episodes}.json",
        ),
        "a",
    ) as file:
        file.write(dumps(Metric(start_time, time(), nb_episodes, game.stats).get()))


if __name__ == "__main__":
    train()
