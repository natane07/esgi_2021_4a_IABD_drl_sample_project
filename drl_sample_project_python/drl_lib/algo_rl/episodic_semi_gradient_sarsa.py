import tqdm

from ..do_not_touch.contracts import DeepSingleAgentWithDiscreteActionsEnv
import tensorflow as tf
import numpy as np
from ..utils import graph_score_bar, graph_score

def episodic_semi_gradient_sarsa(env: DeepSingleAgentWithDiscreteActionsEnv, epsilon, gamma, max_iter, model, name_env=""):
    pre_warm = 10
    state_description_length = env.state_description_length()
    max_actions_count = env.max_actions_count()

    # pour les graph
    score = []
    moyenne = 0.0
    iteration_score = 0
    win = 0
    loss = 0

    for episode_id in tqdm.tqdm(range(max_iter)):
        env.reset()

        while not env.is_game_over():
            s = env.state_description()
            available_actions = env.available_actions_ids()

            if episode_id < pre_warm or np.random.uniform(0.0, 1.0) < epsilon:
                chosen_action = np.random.choice(available_actions)
            else:
                all_q_inputs = np.zeros((len(available_actions), state_description_length + max_actions_count))
                for i, a in enumerate(available_actions):
                    all_q_inputs[i] = np.hstack([s, tf.keras.utils.to_categorical(a, max_actions_count)])

                all_q_values = np.squeeze(model.predict(all_q_inputs))
                chosen_action = available_actions[np.argmax(all_q_values)]
                chosen_action_q_value = np.max(all_q_values)

            previous_score = env.score()
            env.act_with_action_id(chosen_action)
            r = env.score() - previous_score
            s_p = env.state_description()

            if env.is_game_over():
                target = r
                q_inputs = np.hstack([s, tf.keras.utils.to_categorical(chosen_action, max_actions_count)])
                model.train_on_batch(np.array([q_inputs]), np.array([target]))
                break

            next_available_actions = env.available_actions_ids()

            if episode_id < pre_warm or np.random.uniform(0.0, 1.0) < epsilon:
                next_chosen_action = np.random.choice(next_available_actions)
            else:
                next_chosen_action = None
                next_chosen_action_q_value = None
                for a in next_available_actions:
                    q_inputs = np.hstack([s_p, tf.keras.utils.to_categorical(a, max_actions_count)])
                    q_value = model.predict(np.array([q_inputs]))[0][0]
                    if next_chosen_action is None or next_chosen_action_q_value < q_value:
                        next_chosen_action = a
                        next_chosen_action_q_value = q_value

            next_q_inputs = np.hstack([s_p, tf.keras.utils.to_categorical(next_chosen_action, max_actions_count)])
            next_chosen_action_q_value = model.predict(np.array([next_q_inputs]))[0][0]

            target = r + gamma * next_chosen_action_q_value

            q_inputs = np.hstack([s, tf.keras.utils.to_categorical(chosen_action, max_actions_count)])
            model.train_on_batch(np.array([q_inputs]), np.array([target]))

        # Pour les graphs score
        if env.score() == 1 :
            win += 1
        elif env.score() == 0:
            loss += 1
        # print(env.score())
        moyenne = (moyenne * iteration_score + env.score()) / (iteration_score + 1)
        iteration_score += 1
        if episode_id % 10 == 0 and episode_id != 0:
            score.append(moyenne)
            moyenne = 0.0
            iteration_score = 0

    # génération des graphes
    graph_score("episodic_semi_gradient_sarsa on " + name_env, score, 10)
    graph_score_bar("episodic_semi_gradient_sarsa on " + name_env + " - Score des parties pour " + str(max_iter) + " parties jouer", [win, loss])

    return model