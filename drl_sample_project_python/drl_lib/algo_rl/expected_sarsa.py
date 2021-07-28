import numpy as np
from tqdm import *
from ..do_not_touch.contracts import SingleAgentEnv
from ..do_not_touch.result_structures import PolicyAndActionValueFunction
from ..utils import graph_score_bar, graph_score

def expected_sarsa(env: SingleAgentEnv, alpha: float, epsilon: float, gamma: float, max_iter: int, name_env="") -> PolicyAndActionValueFunction:
    assert(epsilon > 0)

    pi = {}
    b = {}
    q = {}

    # pour les graph
    score = []
    moyenne = 0.0
    iteration_score = 0
    win = 0
    loss = 0

    for episode in tqdm(range(max_iter)):
        env.reset()

        while not env.is_game_over():
            s = env.state_id()
            available_actions = env.available_actions_ids()
            if s not in pi:
                pi[s] = {}
                q[s] = {}
                b[s] = {}
                for a in available_actions:
                    pi[s][a] = 1.0 / len(available_actions)
                    q[s][a] = 0.0
                    b[s][a] = 1.0 / len(available_actions)

            available_actions_count = len(available_actions)
            optimal_a = list(q[s].keys())[np.argmax(list(q[s].values()))]
            for a_key, q_s_a in q[s].items():
                if a_key == optimal_a:
                    b[s][a_key] = 1 - epsilon + epsilon / available_actions_count
                else:
                    b[s][a_key] = epsilon / available_actions_count

            chosen_action = np.random.choice(list(b[s].keys()), 1, False, p=list(b[s].values()))[0]
            old_score = env.score()
            env.act_with_action_id(chosen_action)
            r = env.score() - old_score
            s_p = env.state_id()
            next_available_action = env.available_actions_ids()

            if env.is_game_over():
                q[s][chosen_action] += alpha * (r + 0.0 - q[s][chosen_action])
            else:
                if s_p not in pi:
                    pi[s_p] = {}
                    q[s_p] = {}
                    b[s_p] = {}
                    for a in next_available_action:
                        pi[s_p][a] = 1.0 / len(next_available_action)
                        q[s_p][a] = 0.0
                        b[s_p][a] = 1.0 / len(next_available_action)

                somme = 0.0
                for a in next_available_action:
                    somme += pi[s_p][a] * q[s_p][a]
                q[s][chosen_action] += alpha * (r + gamma * somme - q[s][chosen_action])

        # Pour les graphs score
        if env.score() == 1 :
            win += 1
        elif env.score() == 0:
            loss += 1
        # print(env.score())
        moyenne = (moyenne * iteration_score + env.score()) / (iteration_score + 1)
        iteration_score += 1
        if episode % 500 == 0 and episode != 0:
            score.append(moyenne)
            moyenne = 0.0
            iteration_score = 0

    for s in q.keys():
        optimal_a_t = list(q[s].keys())[np.argmax(list(q[s].values()))]
        for a_key, q_s_a in q[s].items():
            if a_key == optimal_a_t:
                pi[s][a_key] = 1.0
            else:
                pi[s][a_key] = 0.0

    # génération des graphes
    graph_score("Expected sarsa on " + name_env, score, 500)
    graph_score_bar("Expected sarsa on "
                    + name_env
                    + " - Score des parties pour "
                    + str(max_iter)
                    + " parties jouées "
                    + "pour alpha = " + str(alpha)
                    + ", epsilon = " + str(epsilon)
                    + ", gamma = " + str(gamma), [win, loss])

    return PolicyAndActionValueFunction(pi=pi, q=q)
