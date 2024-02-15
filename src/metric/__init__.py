class Metric:
    def __init__(
        self: "Metric",
        start_time: float,
        end_time: float,
        nb_episodes: int,
        stats: dict,
    ):
        self.start_time: float = start_time
        self.end_time: float = end_time
        self.nb_episodes: int = nb_episodes
        self.stats: dict = stats

    def get(self: "Metric") -> dict:
        return {
            "duration": self.end_time - self.start_time,
            "loss": self.stats["losses"],
            "mean": {
                "step": self.stats["nb_step"] / self.nb_episodes,
                "time": {
                    "play": (self.end_time - self.start_time) / self.nb_episodes,
                    "step": sum(self.stats["step_times"]) / self.stats["nb_step"],
                }
            },
            "nb_episodes": self.nb_episodes,
            "reward": self.stats["reward"],
            "score": {
                player: {
                    "loose": sum(self.stats["wins"].values()) - value,
                    "mean": (value / sum(self.stats["wins"].values())) * 100,
                    "win": value,
                } for player, value in self.stats["wins"].items()
            },
        }
