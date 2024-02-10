from time import time
from torch.nn import MSELoss
from torch.optim import Adam

from src.agent.cant_stop.double_deep_q_learning_with_experience_replay import DoubleDeepQLearningWithExperienceReplay as DDQLAgent
from src.agent.cant_stop.random import Random as RandomAgent
from src.agent.tool.replay_buffer import ReplayBuffer
from src.config import folder_paths, nb_columns, nb_episodes
from src.entity.cant_stop.color import Color
from src.entity.cant_stop.player import Player
from src.env.cant_stop import CantStop
from src.network.deep_q import DeepQNet


replay_buffer_size = 10_000
batch_size = 64
target_update_frequency = 100
learning_rate = 0.01
gamma = 0.99
epsilon_start = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01

# Initialisation de l'agent avec Replay Buffer
agent = DDQLAgent(
    batch_size=batch_size,
    criterion=MSELoss(),
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
    win_stat: dict = {player.id: 0 for player in game.players}
    start_time=time()

    for _ in range(nb_episodes):
        game.play()
        win_stat[game.won_by.id] += 1
        game.reset()

    agent.model.save(
        folder_paths["cant_stop"],
        "double_deep_q_learning_with_experience_replay.pth",
    )
    end_time=time()

    print(f"Le temps d'exécution de la boucle est de {end_time - start_time} secondes.")
    print(f"Stats: {win_stat}")


if __name__ == "__main__":
    train()
