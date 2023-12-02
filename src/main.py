from env.cant_stop import CantStop
from model.env.cant_stop.color import Color
from model.env.cant_stop.player import Player


players = [
    Player(
        color=(color := Color(name="red")),
        id=1,
        name="Hugo",
    ),
    Player(
        color=(color := Color(name="green")),
        id=2,
        name="Sofianne",
    ),
]

game = CantStop(11, players)
game.play()
