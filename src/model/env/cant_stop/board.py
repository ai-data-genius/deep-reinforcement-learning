from typing import List

from pydantic import BaseModel

from enumerator.env.cant_stop.box import Box as BoxEnum
from model.env.cant_stop.box import Box as BoxModel
from model.env.cant_stop.player import Player
from model.env.cant_stop.way import Way


class Board(BaseModel):
    median_way_id: int
    nb_ways: int
    ways: List[Way] = []

    def generate_way(self, _id: int) -> Way:
        return Way(
            boxes=[
                BoxModel(id=i)
                for i in range(
                    0,
                    (
                        (2 * self.median_way_id - 1) - 2 * (_id - self.median_way_id),
                        2 * _id - 1,
                    )[_id <= self.median_way_id],
                )
            ],
            id=_id,
        )

    def generate_ways(self: 'Board') -> List[Way]:
        return [self.generate_way(_id) for _id in range(2, self.nb_ways+2)]

    def display(self: 'Board'):
        max_width: int = len(self.ways[-1].boxes)  # La longueur de la dernière rangée

        for way in self.ways:
            num_boxes = len(way.boxes)
            spaces = ' ' * (max_width - num_boxes) * 3  # 3 espaces par différence de boîte

            print(f"{way.id:<3}{spaces}", end='')  # Imprimer l'ID du "Way" avec espaces

            for box in way.boxes:
                if box.is_occupied:
                    # Reset à la couleur par défaut après
                    print(
                        f"{box.who_occupies.color.get_ansi_color_code()}[ {box.occupant_type.get_icon()} ]\033[0m",
                        end='  ',
                    )
                elif box.way_won:
                    # Reset à la couleur par défaut après
                    print(f"{box.won_by.color.get_ansi_color_code()}[ ▀ ]\033[0m", end='  ')
                else:
                    print(BoxEnum.EMPTY.value, end='  ')  # Espaces à l'intérieur des crochets

            print()  # Nouvelle ligne après chaque rangée
