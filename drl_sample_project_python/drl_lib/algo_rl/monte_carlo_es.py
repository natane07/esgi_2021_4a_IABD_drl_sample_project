import numpy as np
from tqdm import *
import matplotlib.pyplot as plt

from ..do_not_touch.contracts import SingleAgentEnv
from ..do_not_touch.result_structures import PolicyAndActionValueFunction


def monte_carlo_es(env: SingleAgentEnv,
                   gamma: float,
                   max_iter: int) -> PolicyAndActionValueFunction:
    pi = {}
    q = {}
    returns = {}

    # pour les graph
    score = []
    moyenne = 0.0
    iteration_score = 0
    win = 0
    loss = 0
    egalite = 0

    for episode in tqdm(range(max_iter)):
        env.reset_random()

        # Generate an episode from starting state
        S = []
        A = []
        R = []
        while not env.is_game_over():
            s = env.state_id()
            S.append(s)
            available_actions = env.available_actions_ids()
            if s not in pi:
                pi[s] = {}
                q[s] = {}
                returns[s] = {}
                for a in available_actions:
                    pi[s][a] = 1.0 / len(available_actions)
                    q[s][a] = 0.0
                    returns[s][a] = 0

            chosen_action = np.random.choice(available_actions, 1, False)[0]

            A.append(chosen_action)
            old_score = env.score()
            env.act_with_action_id(chosen_action)
            r = env.score() - old_score
            R.append(r)
        G = 0

        for t in reversed(range(len(S))):
            G = gamma * G + R[t]
            s_t = S[t]
            a_t = A[t]
            found = False
            for s, a in zip(S[:t], A[:t]):
                if s_t == s and a_t == a:
                    found = True
                    break
            if not found:
                q[s_t][a_t] = (q[s_t][a_t] * returns[s_t][a_t] + G) / (returns[s_t][a_t] + 1)
                returns[s_t][a_t] += 1
                pi[s_t] = list(q[s_t].keys())[np.argmax(list(q[s_t].values()))]


        # Pour les graphs scores

        if env.score() == 1 :
            win += 1
        elif env.score() == -1:
            loss += 1
        moyenne = (moyenne * iteration_score + env.score()) / (iteration_score + 1)
        iteration_score += 1
        if episode % 500 == 0 and episode != 0:
            score.append(moyenne)
            moyenne = 0.0
            iteration_score = 0

    # génération des graphes
    graph_score("Monte carlo ES", score, 500)
    graph_score_bar("Monte carlo ES - Score des parties pour " + str(max_iter) + " parties jouer", [win, loss])

    return PolicyAndActionValueFunction(pi, q)

def graph_score(title, scores, scale):
    plt.title(title)
    plt.xlabel("Iterations")
    plt.ylabel("Score moyen pour " + str(scale) + " parties")
    plt.plot(np.arange(1, len(scores) + 1), scores)
    plt.show()

def graph_score_bar(title, scores):
    fig = plt.figure(figsize=(10, 5))
    tile_x = ['Win', 'Loss/Egalite']
    plt.bar(tile_x, scores,width=0.4)
    plt.ylabel("Nombre de parties")
    plt.title(title)
    plt.show()