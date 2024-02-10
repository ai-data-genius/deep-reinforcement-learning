from time import time

from torch.optim import Adam

from src.agent.cant_stop.reinforce_with_mean_baseline import ReinforceWithMeanBaseline as Agent
from src.agent.cant_stop.random import Random as RandomAgent
from src.config import folder_paths, nb_columns, nb_episodes
from src.entity.cant_stop.color import Color
from src.entity.cant_stop.player import Player
from src.env.cant_stop import CantStop
from src.network.reinforce import Reinforce as Net


game: CantStop = CantStop(
    nb_ways=nb_columns,
    players=[
        Player(
            agent=(
                agent := Agent(
                    batch_size=10,
                    decay_rate=.9,
                    epsilon=1.0,
                    gamma=.99,
                    is_policy_gradient=True,
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
            name="Reinforce",
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

    agent.model.save(
        folder_paths["cant_stop"],
        "reinforce_with_mean_baseline.pth",
    )
    end_time=time()

    print(f"Le temps d'exécution de la boucle est de {end_time - start_time} secondes.")
    print(f"Stats: {win_stat}")


if __name__ == "__main__":
    train()
