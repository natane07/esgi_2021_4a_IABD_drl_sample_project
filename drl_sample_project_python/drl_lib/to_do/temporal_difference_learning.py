from ..do_not_touch.result_structures import PolicyAndActionValueFunction
from ..do_not_touch.single_agent_env_wrapper import Env3
from ..envs import tictactoe_single_agent

from ..algo_rl import expected_sarsa
from ..algo_rl import sarsa
from ..algo_rl import q_learning

iteration = 100000
nb_entrainement = 0
tic_tac_toe = tictactoe_single_agent.EnvTicTacToeSingleAgent(200)

def sarsa_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    # TODO
    pass


def q_learning_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a Q-Learning algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the optimal greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    result = q_learning.q_learning(tic_tac_toe, 0.7, 0.1, 0.9, iteration, "TicTacToe")
    tic_tac_toe.view()
    return result


def expected_sarsa_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a Expected SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    result = expected_sarsa.expected_sarsa(tic_tac_toe, 0.7, 0.1, 0.9, iteration, "TicTacToe")
    tic_tac_toe.view()
    return result
    pass


def sarsa_on_secret_env3() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env3
    Launches a SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env3()
    # TODO
    pass


def q_learning_on_secret_env3() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env3
    Launches a Q-Learning algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the optimal greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env3()
    result = q_learning.q_learning(env, 0.3, 0.1, 0.9, iteration, "SecretEnv3")
    return result


def expected_sarsa_on_secret_env3() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env3
    Launches a Expected SARSA Algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the optimal epsilon-greedy Policy and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env3()
    result = expected_sarsa.expected_sarsa(env, 0.3, 0.1, 0.7, iteration, "SecretEnv3")
    return result


def demo():
    print("sarsa_on_tic_tac_toe_solo")
    print(sarsa_on_tic_tac_toe_solo())

    print("q_learning_on_tic_tac_toe_solo")
    print(q_learning_on_tic_tac_toe_solo())

    print("expected_sarsa_on_tic_tac_toe_solo")
    print(expected_sarsa_on_tic_tac_toe_solo())

    print("sarsa_on_secret_env3")
    print(sarsa_on_secret_env3())

    print("q_learning_on_secret_env3")
    print(q_learning_on_secret_env3())

    print("expected_sarsa_on_secret_env3")
    print(expected_sarsa_on_secret_env3())
