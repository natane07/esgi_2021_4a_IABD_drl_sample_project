import numpy as np
from tqdm import *

from ..do_not_touch.result_structures import ValueFunction, PolicyAndValueFunction
from ..do_not_touch.contracts import MDPEnv
from ..utils import graph_value_function
import time

def policy_evaluation(env: MDPEnv, theta: float, gamma: float, env_name='') -> ValueFunction:
    start_time = time.time()
    pi = np.ones((len(env.states()), len(env.actions())))
    pi /= len(env.actions())
    V = np.zeros((len(env.states())))

    while True:
        delta = 0
        for s in env.states():
            v = V[s]
            V[s] = 0.0
            for a in env.actions():
                for s_next in env.states():
                    for r_idx, r in enumerate(env.rewards()):
                        V[s] += pi[s, a] * env.transition_probability(s, a, s_next, r_idx) * (
                                r + gamma * V[s_next])
            delta = max(delta, abs(v - V[s]))

        if delta < theta:
            break

    if env_name == 'Line World':
        graph_value_function("Policy evaluation on " + env_name, V, 1)
    elif env_name == 'Grid World':
        graph_value_function("Policy evaluation on " + env_name, V, 5)

    return dict(enumerate(V.flatten(), 1))


def policy_iteration(env: MDPEnv, theta: float, gamma: float, env_name='') -> PolicyAndValueFunction:
    start_time = time.time()
    V = np.zeros(len(env.states()))
    pi = np.ones((len(env.states()), len(env.actions())))
    pi /= len(env.actions())
    res_pi = {}


    while True:
        while True:
            delta = 0
            for s in env.states():
                v = V[s]
                V[s] = 0.0
                for a in env.actions():
                    for s_next in env.states():
                        for r_idx, r in enumerate(env.rewards()):
                            V[s] += pi[s, a] * env.transition_probability(s, a, s_next, r_idx) * (
                                    r + gamma * V[s_next])
                delta = max(delta, abs(v - V[s]))

            if delta < theta:
                break

        policy_stable = True
        for s in env.states():
            old_action = pi[s, :]
            best_a = None
            best_a_value = None
            for a in env.actions():
                a_value = 0
                for s_next in env.states():
                    for r_index, r in enumerate(env.rewards()):
                        a_value += env.transition_probability(s, a, s_next, r_index) * (r + gamma * V[s_next])

                if best_a_value is None or best_a_value < a_value:  # argmax
                    best_a_value = a_value
                    best_a = a

            pi[s, :] = 0.0
            pi[s, best_a] = 1.0
            if not np.array_equal(pi[s], old_action):
                print(s)
                policy_stable = False

        if policy_stable:
            break

    for index, value in enumerate(pi):
        res_pi[index] = dict(enumerate(value.flatten(), 1))

    if env_name == 'Line World':
        graph_value_function("Policy iteration on " + env_name, V, 1)
    elif env_name == 'Grid World':
        graph_value_function("Policy iteration on " + env_name, V, 5)

    return PolicyAndValueFunction(res_pi, dict(enumerate(V.flatten(), 1)))


def value_iteration(env: MDPEnv, theta: float, gamma: float, env_name='') -> PolicyAndValueFunction:
    start_time = time.time()
    V = np.zeros((len(env.states())))
    pi = np.ones((len(env.states()), len(env.actions())))
    pi /= len(env.actions())
    pi_else = pi.copy()
    res_pi = {}
    wins = 0
    losses = 0

    while True:
        delta = 0
        for s in env.states():
            v = V[s]
            V[s] = 0.0
            best_a = None
            best_a_value = None
            for a in env.actions():
                a_value = 0
                for s_next in env.states():
                    for r_index, r in enumerate(env.rewards()):
                        tmp_a_value = env.transition_probability(s, a, s_next, r_index) * (r + gamma * V[s_next])
                        a_value += tmp_a_value
                        V[s] += pi[s, a] * tmp_a_value

                if best_a_value is None or best_a_value < a_value:  # argmax
                    best_a_value = a_value
                    best_a = a

            delta = max(delta, abs(V[s] - v))
            pi_else[s, :] = 0.0
            pi_else[s, best_a] = 1.0

        if delta < theta:
            break

        if env.is_state_terminal(s):
            if V[len(V) - 2] == 1.0:
                wins += 1
            else:
                losses += 1

        for index, value in enumerate(pi_else):
            res_pi[index] = dict(enumerate(value.flatten(), 1))

    if env_name == 'Line World':
        graph_value_function("Value iteration on " + env_name, V, 1)
    elif env_name == 'Grid World':
        graph_value_function("Value iteration on " + env_name, V, 5)

    return PolicyAndValueFunction(res_pi, dict(enumerate(V.flatten(), 1)))
