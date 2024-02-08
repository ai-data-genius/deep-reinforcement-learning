from time import time

from torch.nn import MSELoss
from torch.optim import Adam

from src.agent.cant_stop.deep_q_learning import DeepQLearning as DQLAgent
from src.agent.cant_stop.random import Random as RandomAgent
from src.agent.tool.replay_buffer import ReplayBuffer
from src.entity.cant_stop.color import Color
from src.entity.cant_stop.player import Player
from src.env.cant_stop import CantStop
from src.network.deep_q import DeepQNet


nb_columns: int = 11

agent: DQLAgent = DQLAgent(
    batch_size=10,
    criterion=MSELoss(),
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

win_stat: dict = {player.id: 0 for player in game.players}
start_time=time()

for i in range(100):
    game.play()
    win_stat[game.won_by.id] += 1
    game.reset()

end_time=time()

print(f"Le temps d'exécution de la boucle est de {end_time - start_time} secondes.")
print(f"Stats: {win_stat}")
