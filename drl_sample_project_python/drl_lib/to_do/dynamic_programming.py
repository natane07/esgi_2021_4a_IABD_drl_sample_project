import numpy as np

from ..do_not_touch.mdp_env_wrapper import Env1
from ..do_not_touch.result_structures import ValueFunction, PolicyAndValueFunction
from ..envs.line_world_env import LineWorld
from ..envs.grid_world_env import GridWorld
from ..algo_rl.dynamic_programming import policy_evaluation, policy_iteration, value_iteration


def policy_evaluation_on_line_world() -> ValueFunction:
    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Evaluation Algorithm in order to find the Value Function of a uniform random policy
    Returns the Value function (V(s)) of this policy
    """
    line_world = LineWorld(7)
    theta = 0.001
    gamma = 1.0

    return policy_evaluation(line_world, theta, gamma)


def policy_iteration_on_line_world() -> PolicyAndValueFunction:
    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    line_world = LineWorld(7)
    theta = 0.001
    gamma = 1.0

    return policy_iteration(line_world, theta, gamma)


def value_iteration_on_line_world() -> PolicyAndValueFunction:
    """
    Creates a Line World of 7 cells (leftmost and rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Value Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    line_world = LineWorld(7)
    theta = 0.001
    gamma = 1.0

    return value_iteration(line_world, theta, gamma)


def policy_evaluation_on_grid_world() -> ValueFunction:
    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Evaluation Algorithm in order to find the Value Function of a uniform random policy
    Returns the Value function (V(s)) of this policy
    """
    grid_world = GridWorld(5, 5)
    theta = 0.001
    gamma = 1.0

    return policy_evaluation(grid_world, theta, gamma)
    pass


def policy_iteration_on_grid_world() -> PolicyAndValueFunction:
    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Policy Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    grid_world = GridWorld(5, 5)
    gamma = 1.0
    theta = 0.001

    return policy_iteration(grid_world, theta, gamma)
    pass


def value_iteration_on_grid_world() -> PolicyAndValueFunction:
    """
    Creates a Grid World of 5x5 cells (upper rightmost and lower rightmost are terminal, with -1 and 1 reward respectively)
    Launches a Value Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    grid_world = GridWorld(5, 5)
    theta = 0.001
    gamma = 1.0

    return value_iteration(grid_world, theta, gamma)
    pass


def policy_evaluation_on_secret_env1() -> ValueFunction:
    """
    Creates a Secret Env1
    Launches a Policy Evaluation Algorithm in order to find the Value Function of a uniform random policy
    Returns the Value function (V(s)) of this policy
    """
    env = Env1()
    theta = 0.001
    gamma = 1.0

    return policy_evaluation(env, theta, gamma)


def policy_iteration_on_secret_env1() -> PolicyAndValueFunction:
    """
    Creates a Secret Env1
    Launches a Policy Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Returns the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    env = Env1()
    gamma = 1.0
    theta = 0.001

    return policy_iteration(env, theta, gamma)


def value_iteration_on_secret_env1() -> PolicyAndValueFunction:
    """
    Creates a Secret Env1
    Launches a Value Iteration Algorithm in order to find the Optimal Policy and its Value Function
    Prints the Policy (Pi(s,a)) and its Value Function (V(s))
    """
    env = Env1()
    gamma = 1.0
    theta = 0.001

    return value_iteration(env, theta, gamma)


def demo():
    print(policy_evaluation_on_line_world())
    print(policy_iteration_on_line_world())
    print(value_iteration_on_line_world())

    print(policy_evaluation_on_grid_world())
    print(policy_iteration_on_grid_world())
    print(value_iteration_on_grid_world())

    print(policy_evaluation_on_secret_env1())
    print(policy_iteration_on_secret_env1())
    print(value_iteration_on_secret_env1())
