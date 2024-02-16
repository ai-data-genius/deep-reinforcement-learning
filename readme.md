# Data Genius - Deep Reinforcement Learning

The Deep Reinforcement Learning project, led by Nicolas Vidal, focuses on the evaluation of different reinforcement learning techniques. We will be working on different environments such as Line World, Grid World, Tic Tac Toe, Cant Stop and Balloon Pop, applying learning algorithms such as Deep Q-Learning, REINFORCE, and MuZero. Objectives include understanding the strengths of each algorithm and their appropriate applications. Deliverables include a detailed code base, trained models and a report demonstrating methodology, results and interpretations. The project focuses on the theoretical understanding and practical application of reinforcement learning concepts.

# Comment exécuter le code ?

Tout d'abord faire : `pip install -e .`.
Puis installer les libs : torch, numpy, matplotlib, pydantic et click.
Puis pour jour une partie contre l'ordinateur faire : `python src/play.py `.
Pour entrainer un agent faire : `python src/main.py train --agent="{MON_AGENT}"`.
Pour faire prédire un agent faire : `python src/predict/{MON_ENV}/{MON_AGENT}.py`.
Pour voir des métriques faire : `python src/main.py show_metrics --file_name="{MON_FICHIER_DE_METRIQUES}"`, vous trouvez votre fichier de métriques dans ./metrics/{MON_ENV}/.
