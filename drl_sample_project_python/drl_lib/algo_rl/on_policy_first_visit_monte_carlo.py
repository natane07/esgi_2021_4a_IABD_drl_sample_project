import numpy as np
from tqdm import *
from ..do_not_touch.contracts import SingleAgentEnv
from ..do_not_touch.result_structures import PolicyAndActionValueFunction
from ..utils import graph_score_bar, graph_score

def on_policy_first_visit_monte_carlo_control(
        env: SingleAgentEnv,
        gamma: float,
        eps: float,
        max_iter: int
) -> PolicyAndActionValueFunction:
    assert(eps > 0)
    pi = {}
    q = {}
    returns = {}

    # pour les graph
    score = []
    moyenne = 0.0
    iteration_score = 0
    win = 0
    loss = 0

    for episode in tqdm(range(max_iter)):
        env.reset_random()

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

            chosen_action = np.random.choice(
                list(pi[s].keys()),
                1,
                False,
                p=list(pi[s].values())
            )[0]

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
                optimal_s_a = list(q[s_t].keys())[np.argmax(list(q[s_t].values()))]
                available_action_t_counts = len(q[s_t])
                for a in q[s_t].keys():
                    if a == optimal_s_a:
                        pi[s_t][a] = 1 - eps + eps/available_action_t_counts
                    else:
                        pi[s_t][a] = eps / available_action_t_counts

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
    graph_score("On policy first visit Monte carlo", score, 500)
    graph_score_bar("On policy first visit Monte carlo - Score des parties pour " + str(max_iter) + " parties jouer", [win, loss])

    return PolicyAndActionValueFunction(pi, q)