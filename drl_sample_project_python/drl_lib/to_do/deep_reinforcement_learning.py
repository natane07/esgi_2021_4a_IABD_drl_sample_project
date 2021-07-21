from ..do_not_touch.deep_single_agent_with_discrete_actions_env_wrapper import Env5
import tensorflow as tf
from ..envs import tictactoe_DSA
from ..algo_rl import episodic_semi_gradient_sarsa


tic_tac_toe = tictactoe_DSA.EnvTicTacToeDeepSingleAgent(100)
iteration = 100

class DeepQNetwork:
    """
    Contains the weights, structure of the q_network
    """
    # TODO
    pass


def episodic_semi_gradient_sarsa_on_tic_tac_toe_solo() -> DeepQNetwork:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a episodic semi gradient sarsa Algorithm in order to find the optimal epsilon-greedy action_value function
    Returns the optimal epsilon-greedy action_value function (Q(w)(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    state_description_length = tic_tac_toe.state_description_length()
    max_actions_count = tic_tac_toe.max_actions_count()
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation=tf.keras.activations.tanh,
                              input_dim=(state_description_length + max_actions_count)),
        tf.keras.layers.Dense(64, activation=tf.keras.activations.tanh),
        tf.keras.layers.Dense(128, activation=tf.keras.activations.tanh),
        tf.keras.layers.Dense(256, activation=tf.keras.activations.tanh),
        tf.keras.layers.Dense(128, activation=tf.keras.activations.tanh),
        tf.keras.layers.Dense(1, activation=tf.keras.activations.linear),
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.mse)
    result = episodic_semi_gradient_sarsa.episodic_semi_gradient_sarsa(tic_tac_toe, 0.1, 0.9, iteration, model, "TicTacToe")
    result.save('./drl_lib/model_tf/episodic_semi_gradient_sarsa_tic_tac_toe.h5')

def deep_q_learning_on_tic_tac_toe_solo() -> DeepQNetwork:
    """
    Creates a TicTacToe Solo environment (Single player versus Uniform Random Opponent)
    Launches a deep q Learning (DQN) algorithm in order to find the optimal action_value function
    Returns the optimal action_value function (Q(w)(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    # TODO
    pass


def episodic_semi_gradient_sarsa_on_pac_man() -> DeepQNetwork:
    """
    Creates a PacMan environment
    Launches a episodic semi gradient sarsa Algorithm in order to find the optimal epsilon-greedy action_value function
    Returns the optimal epsilon-greedy action_value function (Q(w)(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    # TODO
    pass


def deep_q_learning_on_pac_man() -> DeepQNetwork:
    """
    Creates a PacMan environment
    Launches a deep q Learning (DQN) algorithm in order to find the optimal action_value function
    Returns the optimal action_value function (Q(w)(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    # TODO
    pass


def episodic_semi_gradient_sarsa_on_secret_env5() -> DeepQNetwork:
    """
    Creates a Secret Env 5 environment
    Launches a episodic semi gradient sarsa Algorithm in order to find the optimal epsilon-greedy action_value function
    Returns the optimal epsilon-greedy action_value function (Q(w)(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    env = Env5()
    state_description_length = env.state_description_length()
    max_actions_count = env.max_actions_count()
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation=tf.keras.activations.tanh,
                              input_dim=(state_description_length + max_actions_count)),
        tf.keras.layers.Dense(64, activation=tf.keras.activations.tanh),
        tf.keras.layers.Dense(128, activation=tf.keras.activations.tanh),
        tf.keras.layers.Dense(256, activation=tf.keras.activations.tanh),
        tf.keras.layers.Dense(128, activation=tf.keras.activations.tanh),
        tf.keras.layers.Dense(1, activation=tf.keras.activations.linear),
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.mse)
    result = episodic_semi_gradient_sarsa.episodic_semi_gradient_sarsa(env, 0.1, 0.9, iteration, model, "SecretEnv5")
    result.save('./drl_lib/model_tf/episodic_semi_gradient_sarsa_secretenv5.h5')


def deep_q_learning_on_secret_env5() -> DeepQNetwork:
    """
    Creates a Secret Env 5 environment
    Launches a deep q Learning (DQN) algorithm in order to find the optimal action_value function
    Returns the optimal action_value function (Q(w)(s,a))
    Experiment with different values of hyper parameters and choose the most appropriate combination
    """
    # TODO
    _env = Env5()
    pass


def demo():
    print("episodic_semi_gradient_sarsa_on_tic_tac_toe_solo")
    print(episodic_semi_gradient_sarsa_on_tic_tac_toe_solo())

    print("deep_q_learning_on_tic_tac_toe_solo")
    print(deep_q_learning_on_tic_tac_toe_solo())

    print("episodic_semi_gradient_sarsa_on_pac_man")
    print(episodic_semi_gradient_sarsa_on_pac_man())

    print("deep_q_learning_on_pac_man")
    print(deep_q_learning_on_pac_man())

    print("episodic_semi_gradient_sarsa_on_secret_env5")
    print(episodic_semi_gradient_sarsa_on_secret_env5())

    print("deep_q_learning_on_secret_env5")
    print(deep_q_learning_on_secret_env5())

