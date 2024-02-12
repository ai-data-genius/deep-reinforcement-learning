from os.path import abspath
from subprocess import CalledProcessError, PIPE, run

from click import command, group, option


@group()
def cli():
    pass


@command()
@option("--game", default="cant_stop", help="Env name you would like to train.")
@option("--agent", default="human", help="Agent name you would like to train.")
def train(game: str, agent: str):
    agent = {
        "dql": "deep_q_learning",
        "ddql": "double_deep_q_learning",
        "ddql_exp": "double_deep_q_learning_with_experience_replay",
        "ddql_prio_exp": "double_deep_q_learning_with_prioritized_experience_replay",
        "reinforce": "reinforce",
        "reinforce_mb": "reinforce_with_mean_baseline",
        "reinforce_blc": "reinforce_with_baseline_learned_by_a_critic",
        "mcts": "monte_carlo_tree_search",
    }[agent]

    try:
        print(
            run(
                [
                    "python",
                    f"{abspath('.')}/src/train/{game}/{agent}.py",
                ],
                check=True,
                stdout=PIPE,
                stderr=PIPE,
            )
            .stdout
            .decode('utf-8')
        )
    except CalledProcessError as e:
        print("Erreur lors de l'exécution du script :")
        print(e.stderr.decode())
    except Exception as e:
        print("Autre erreur :")
        print(e.stderr.decode())


cli.add_command(train)


if __name__ == '__main__':
    cli()
