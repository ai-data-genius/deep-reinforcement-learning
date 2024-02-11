from os.path import abspath
from typing import Dict


folder_paths: Dict[str, str] = {
    "cant_stop": f"{abspath('.')}/models/cant_stop/",
    "balloon_pop": f"{abspath('.')}/models/balloon_pop/",
}

nb_columns: int = 11
nb_episodes: int = 100
