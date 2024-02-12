from time import time

from torch.optim import Adam

from src.agent.cant_stop.ppo import PPO as Agent
from src.agent.cant_stop.random import Random as RandomAgent
from src.agent.tool.ppo_replay_buffer import PPOReplayBuffer
from src.config import folder_paths, nb_columns, nb_episodes
from src.entity.cant_stop.color import Color
from src.entity.cant_stop.player import Player
from src.env.cant_stop import CantStop
from src.network.ppo import PPO as Net


game: CantStop = CantStop(
    nb_ways=nb_columns,
    players=[
        Player(
            agent=(
                agent := Agent(
                    batch_size=10,
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
    win_stat: dict = {player.id: 0 for player in game.players}
    start_time=time()

    for _ in range(nb_episodes):
        game.play()
        win_stat[game.won_by.id] += 1
        game.reset()

    agent.model.save(folder_paths["cant_stop"], "ppo.pth")
    end_time=time()

    print(f"Le temps d'exécution de la boucle est de {end_time - start_time} secondes.")
    print(f"Stats: {win_stat}")


if __name__ == "__main__":
    train()
