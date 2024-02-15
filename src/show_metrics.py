import json
import matplotlib.pyplot as plt
from os.path import join

from numpy import mean

from config import folder_paths


def plot_metrics(player: str, value: list, losses: list) -> None:
    episode_range = range(1, len(value) + 1)

    # Plot des récompenses cumulatives par épisode
    fig, ax1 = plt.subplots()
    ax1.set_xlabel("Épisode")
    ax1.set_ylabel("Récompense cumulative par épisode", color="tab:blue")
    ax1.plot(episode_range, [mean(value[max(0, i - 100):i]) for i in episode_range], color='tab:blue')
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    if losses != []:
        ax2 = ax1.twinx()
        ax2.set_ylabel('Perte (moyenné sur 100 épisodes)', color='tab:red')
        ax2.plot(episode_range, [mean(losses[max(0, i - 100):i]) for i in episode_range], color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')

    fig.tight_layout()
    plt.title(f"Évolution de l'agent {player}")
    plt.show()


def plot_pie(score: dict) -> None:
    plt.title(f"Win rates")
    plt.pie(
        [player["mean"] for player in score.values()],
        labels=[player for player in score],
        colors=['gold','lightcoral'],
        autopct='%1.1f%%',
        shadow=True,
        startangle=140,
    )
    plt.show()


def get_plot(file_name: str) -> None:
    # Charger les métriques
    with open(join(folder_paths["metrics"]["cant_stop"], f"{file_name}.json"), 'r') as file:
        metrics = json.load(file)

    for player, value in metrics["reward"].items():
        plot_metrics(player, value, metrics["loss"][player])

    plot_pie(metrics["score"])
