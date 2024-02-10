from src.agent.cant_stop.human import Human
from src.entity.cant_stop.color import Color
from src.entity.cant_stop.player import Player
from src.env.cant_stop import CantStop


CantStop(
    11,
    [
        Player(
            agent=Human(),
            color=Color(name="red"),
            id=1,
            name="Hugo",
        ),
        Player(
            agent=Human(),
            color=Color(name="green"),
            id=2,
            name="David",
        ),
    ],
).play(True)
