from entity.env.cant_stop.color import Color
from entity.env.cant_stop.player import Player
from env.cant_stop import CantStop


game = CantStop(
    11,
    [
        Player(
            color=(color := Color(name="red")),
            id=1,
            name="Hugo",
            type="human",
        ),
        Player(
            color=(color := Color(name="green")),
            id=2,
            name="David",
            type="human",
        ),
    ],
)
game.play()
