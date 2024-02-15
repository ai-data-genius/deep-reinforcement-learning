from os.path import abspath
from subprocess import CalledProcessError, PIPE, run

from click import command, group, option

from src.config import agents
from src.show_metrics import get_plot

@group()
def cli():
    pass


def _train(game: str, agent: str) -> None:
    try:
        print(
            run(
                [
                    "python",
                    f"{abspath('.')}/src/train/{game}/{agents[agent]}.py",
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


@command("train")
@option("--game", default="cant_stop", help="Env name you would like to train.")
@option("--agent", default="human", help="Agent name you would like to train.")
def train(game: str, agent: str) -> None:
    _train(game, agent)


@command("train_all")
@option("--game", default="cant_stop", help="Env name you would like to train.")
def train_all(game: str) -> None:
    for agent in agents:
        _train(game, agent)


@command("show_metrics")
@option("--file_name", default="", help="Nom du fichier avec les métriques.")
def show_metrics(file_name: str) -> None:
    if file_name == "":
        raise ValueError("Il faut renseigner un filename !")

    get_plot(file_name)


cli.add_command(train)
cli.add_command(train_all)
cli.add_command(show_metrics)


if __name__ == '__main__':
    cli()
