from datetime import datetime
from json import dumps
from os.path import join
from time import time

from torch.nn import SmoothL1Loss  # Huber
from torch.optim import Adam

from src.agent.cant_stop.double_deep_q_learning_with_experience_replay import DoubleDeepQLearningWithExperienceReplay as DDQLAgent
from src.agent.cant_stop.random import Random as RandomAgent
from src.agent.tool.replay_buffer import ReplayBuffer
from src.config import folder_paths, nb_columns, nb_episodes
from src.entity.cant_stop.color import Color
from src.entity.cant_stop.player import Player
from src.env.cant_stop import CantStop
from src.metric import Metric
from src.network.deep_q import DeepQNet


trained_agent: str = "double_deep_q_learning_with_experience_replay"

replay_buffer_size = 10_000
batch_size = 64
target_update_frequency = 100
learning_rate = .0001
gamma = 0.99
epsilon_start = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01

# Initialisation de l'agent avec Replay Buffer
agent = DDQLAgent(
    batch_size=batch_size,
    criterion=SmoothL1Loss(),
    decay_rate=epsilon_decay,
    epsilon=epsilon_start,
    gamma=gamma,
    model=(model := DeepQNet(
        input_size=(nb_columns * 7) + 4,  # Nombre de colonnes * 7 caractéristiques chacune + 4 dés
        hidden_size=28,  # Paramètre à ajuster selon les besoins
        output_size=pow(nb_columns, 2) + 1,  # +1 pour l'action de continuer à jouer
    )),
    memory=ReplayBuffer(replay_buffer_size),
    num_columns=nb_columns,
    optimizer=Adam(model.parameters(), lr=learning_rate),
    replay_buffer_size=replay_buffer_size,
    target_update_frequency=target_update_frequency,
)

game = CantStop(
    nb_ways=nb_columns,
    players=[
        Player(
            agent=agent,
            color=Color(name="red"),
            id=1,
            name="DDQLAgent",
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
    start_time=time()

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
