from os.path import abspath
from typing import Dict


agents: Dict[str, str] = {
    "dql": "deep_q_learning",
    "ddql": "double_deep_q_learning",
    "ddql_exp": "double_deep_q_learning_with_experience_replay",
    "ddql_prio_exp": "double_deep_q_learning_with_prioritized_experience_replay",
    "reinforce": "reinforce",
    "reinforce_mb": "reinforce_with_mean_baseline",
    "reinforce_blc": "reinforce_with_baseline_learned_by_a_critic",
    "mcts": "monte_carlo_tree_search",
    "ppo": "ppo",
}

folder_paths: Dict[str, str] = {
    "metrics": {
        "cant_stop": f"{abspath('.')}/metrics/cant_stop/",
        "balloon_pop": f"{abspath('.')}/metrics/balloon_pop/",
    },
    "models": {
        "cant_stop": f"{abspath('.')}/models/cant_stop/",
        "balloon_pop": f"{abspath('.')}/models/balloon_pop/",
    },
}

nb_columns: int = 11
nb_episodes: int = 100
