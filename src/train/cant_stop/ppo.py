from datetime import datetime
from json import dumps
from os.path import join
from time import time

from torch.optim import Adam

from src.agent.cant_stop.ppo import PPO as Agent
from src.agent.cant_stop.random import Random as RandomAgent
from src.agent.tool.ppo_replay_buffer import PPOReplayBuffer
from src.config import folder_paths, nb_columns, nb_episodes
from src.entity.cant_stop.color import Color
from src.entity.cant_stop.player import Player
from src.env.cant_stop import CantStop
from src.metric import Metric
from src.network.ppo import PPO as Net


trained_agent: str = "ppo"

game: CantStop = CantStop(
    nb_ways=nb_columns,
    players=[
        Player(
            agent=(
                agent := Agent(
                    batch_size=64,
                    decay_rate=.9,
                    eps_clip=.2,
                    gamma=.99,
                    K_epochs=10,
                    memory=PPOReplayBuffer(10_000),
                    model=(model := Net(
                        # le nombre de colonne * 7 caractéristiques chacune + 4 dès
                        input_size=(nb_columns * 7) + 4,
                        hidden_size=128,
                        output_size=pow(nb_columns, 2) + 1,  # + 1 pour keep_playing action
                    )),
                    num_columns=nb_columns,
                    optimizer=Adam(model.parameters(), lr=.01),
                )
            ),
            color=Color(name="red"),
            id=1,
            name="PPO",
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

    agent.model.save(folder_paths["models"]["cant_stop"], f"{trained_agent}.pth")

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
