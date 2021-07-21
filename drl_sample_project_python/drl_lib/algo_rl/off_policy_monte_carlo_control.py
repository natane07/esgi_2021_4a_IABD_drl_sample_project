import numpy as np
from tqdm import *

from ..do_not_touch.contracts import SingleAgentEnv
from ..do_not_touch.result_structures import PolicyAndActionValueFunction
from ..utils import graph_score_bar, graph_score


def off_policy_monte_carlo_control(
        env: SingleAgentEnv,
        gamma: float,
        max_iter: int,
        name_env=""
) -> PolicyAndActionValueFunction:
    q = {}
    c = {}
    pi = {}

    # pour les graph
    score = []
    moyenne = 0.0
    iteration_score = 0
    win = 0
    loss = 0

    def random_soft_policy_for_state(env: SingleAgentEnv, b):
        rng = np.random.default_rng()
        if env.state_id() not in b:
            b[env.state_id()] = {}
            b_s = b[env.state_id()]
            available_actions = env.available_actions_ids()
            nb_of_actions = len(available_actions)
            proba_s = rng.integers(1, nb_of_actions, nb_of_actions, endpoint=True).astype(float)
            proba_s /= sum(proba_s)
            for id_a, a in enumerate(available_actions):
                b_s[a] = proba_s[id_a]
            # print(b_s.values())

    for episode in tqdm(range(max_iter)):
        env.reset()
        b = {}  # dict(dict(float)) soft behavior policy
        S = []
        A = []
        R = []
        while not env.is_game_over():
            s = env.state_id()
            S.append(s)
            random_soft_policy_for_state(env, b)
            available_actions = env.available_actions_ids()
            if s not in q:
                q[s] = {}
                c[s] = {}
                for a in available_actions:
                    q[s][a] = 0.0
                    c[s][a] = 0.0
            chosen_action = np.random.choice(
                list(b[s].keys()),
                1,
                False,
                p=list(b[s].values())
            )[0]

            A.append(chosen_action)
            old_score = env.score()
            env.act_with_action_id(chosen_action)
            r = env.score() - old_score
            R.append(r)

        G = 0.0
        W = 1.0
        for t in reversed(range(len(S))):
            s_t = S[t]
            a_t = A[t]
            G = gamma * G + R[t]
            c[s_t][a_t] += W
            q[s_t][a_t] += (W / c[s_t][a_t] * (G - q[s_t][a_t]))
            pi[s_t] = list(q[s_t].keys())[np.argmax(list(q[s_t].values()))]
            if a_t == pi[s_t]:
                break
            W = W / b[s_t][a_t]

        # Pour les graphs score
        if env.score() == 1 :
            win += 1
        elif env.score() == 0:
            loss += 1
        moyenne = (moyenne * iteration_score + env.score()) / (iteration_score + 1)
        iteration_score += 1
        if episode % 500 == 0 and episode != 0:
            score.append(moyenne)
            moyenne = 0.0
            iteration_score = 0

    # génération des graphes
    graph_score("Off policy Monte carlo Control on " + name_env, score, 500)
    graph_score_bar("Off policy Monte carlo Control on " + name_env + " - Score des parties pour " + str(max_iter) + " parties jouer", [win, loss])

    return PolicyAndActionValueFunction(pi, q)
