from ..do_not_touch.result_structures import PolicyAndActionValueFunction
from ..do_not_touch.single_agent_env_wrapper import Env2
from ..envs import tictactoe_single_agent
from ..algo_rl import monte_carlo_es
from ..algo_rl import on_policy_first_visit_monte_carlo
from ..algo_rl import off_policy_monte_carlo_control

iteration = 100000
nb_entrainement = 0
tic_tac_toe = tictactoe_single_agent.EnvTicTacToeSingleAgent(200)

def monte_carlo_es_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a Monte Carlo ES (Exploring Starts) in order to find the optimal Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    """
    result = monte_carlo_es.monte_carlo_es(tic_tac_toe, 0.99, iteration, "TicTacToe")
    tic_tac_toe.view()
    for _ in range(0, nb_entrainement):
        tic = tictactoe_single_agent.EnvTicTacToeSingleAgent(100, pi=result.pi)
        result = monte_carlo_es.monte_carlo_es(tic, 0.99, iteration, "TicTacToe")
        tic.view()
    return result


def on_policy_first_visit_monte_carlo_control_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches an On Policy First Visit Monte Carlo Control algorithm in order to find the optimal epsilon-greedy Policy
    and its action-value function
    Returns the Optimal epsilon-greedy Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    result = on_policy_first_visit_monte_carlo.on_policy_first_visit_monte_carlo_control(
        tic_tac_toe,
        0.99,
        0.1,
        iteration,
        "TicTacToe"
    )
    tic_tac_toe.view()
    for _ in range(0, nb_entrainement):
        tic = tictactoe_single_agent.EnvTicTacToeSingleAgent(100, pi=result.pi)
        result = on_policy_first_visit_monte_carlo.on_policy_first_visit_monte_carlo_control(
            tic,
            0.99,
            0.1,
            iteration,
            "TicTacToe"
        )
        tic.view()
    return result


def off_policy_monte_carlo_control_on_tic_tac_toe_solo() -> PolicyAndActionValueFunction:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches an Off Policy Monte Carlo Control algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    result = off_policy_monte_carlo_control.off_policy_monte_carlo_control(tic_tac_toe, 0.99, iteration, "TicTacToe")
    tic_tac_toe.view()
    for _ in range(0, nb_entrainement):
        tic = tictactoe_single_agent.EnvTicTacToeSingleAgent(100, pi=result.pi)
        result = off_policy_monte_carlo_control.off_policy_monte_carlo_control(tic, 0.99, iteration, "TicTacToe")
        tic.view()
    return result


def monte_carlo_es_on_secret_env2() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env2
    Launches a Monte Carlo ES (Exploring Starts) in order to find the optimal Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    """
    env = Env2()
    result = monte_carlo_es.monte_carlo_es(env, 0.99, iteration, "SecretEnv2")
    return result


def on_policy_first_visit_monte_carlo_control_on_secret_env2() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env2
    Launches an On Policy First Visit Monte Carlo Control algorithm in order to find the optimal epsilon-greedy Policy and its action-value function
    Returns the Optimal epsilon-greedy Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env2()
    result = on_policy_first_visit_monte_carlo.on_policy_first_visit_monte_carlo_control(env, 0.99, 0.1, iteration, "SecretEnv2")
    return result


def off_policy_monte_carlo_control_on_secret_env2() -> PolicyAndActionValueFunction:
    """
    Creates a Secret Env2
    Launches an Off Policy Monte Carlo Control algorithm in order to find the optimal greedy Policy and its action-value function
    Returns the Optimal Policy (Pi(s,a)) and its Action-Value function (Q(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env2()
    result = off_policy_monte_carlo_control.off_policy_monte_carlo_control(env, 0.99, iteration, "SecretEnv2")
    return result


def demo():
    print("monte_carlo_es_on_tic_tac_toe_solo \n")
    print(monte_carlo_es_on_tic_tac_toe_solo())

    print("on_policy_first_visit_monte_carlo_control_on_tic_tac_toe_solo \n")
    print(on_policy_first_visit_monte_carlo_control_on_tic_tac_toe_solo())

    print("off_policy_monte_carlo_control_on_tic_tac_toe_solo \n")
    print(off_policy_monte_carlo_control_on_tic_tac_toe_solo())

    print("monte_carlo_es_on_secret_env2 \n")
    print(monte_carlo_es_on_secret_env2())

    print("on_policy_first_visit_monte_carlo_control_on_secret_env2 \n")
    print(on_policy_first_visit_monte_carlo_control_on_secret_env2())

    print("off_policy_monte_carlo_control_on_secret_env2 \n")
    print(off_policy_monte_carlo_control_on_secret_env2())
