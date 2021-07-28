import operator
import random
import numpy as np
from ..do_not_touch.contracts import DeepSingleAgentWithDiscreteActionsEnv


class EnvTicTacToeDeepSingleAgent(DeepSingleAgentWithDiscreteActionsEnv):
    def __init__(self, max_steps: int, first_player: int = 0, pi=None):
        assert (max_steps > 0)
        self.max_steps = max_steps
        self.pi = pi
        self.reset()

    def is_game_over(self) -> bool:
        return self.game_over

    def state_description(self) -> np.ndarray:
        return np.array(self.case)

    def state_description_length(self) -> int:
        return len(self.case)

    def max_actions_count(self) -> int:
        return 9

    def state_id(self) -> int:
        num = ''
        for case in self.case:
            if case == 0:
                num += '1'
            elif case == 1:
                num += '2'
            else:
                num += '3'
        return int(num)

    def is_win(self, case=[0, 1, 2], number_win=3):
        if self.case[case[0]] + self.case[case[1]] + self.case[case[2]] == number_win:
            return True
        return False

    def act_with_action_id(self, action_id: int):
        assert (not self.game_over)
        assert (action_id in [0, 1, 2, 3, 4, 5, 6, 7, 8])
        self.case[action_id] = 1

        # Toutes les possibiltés de gagner
        ligne0 = self.is_win(case=[0, 1, 2], number_win=3)
        ligne1 = self.is_win(case=[3, 4, 5], number_win=3)
        ligne2 = self.is_win(case=[6, 7, 8], number_win=3)
        colonne0 = self.is_win(case=[0, 3, 6], number_win=3)
        colonne1 = self.is_win(case=[1, 4, 7], number_win=3)
        colonne2 = self.is_win(case=[2, 5, 8], number_win=3)
        diagonal0 = self.is_win(case=[0, 4, 8], number_win=3)
        diagonal1 = self.is_win(case=[2, 4, 6], number_win=3)

        # Verification si la grille est complete
        grille_complete = True
        for b in self.case:
            if b == 0:
                grille_complete = False

        # Verification si il y a une un gagnant
        if ligne0 or ligne1 or ligne2 or colonne0 or colonne1 or colonne2 or diagonal0 or diagonal1:
            self.game_over = True
            self.current_score = 1.0
            return

        # Si la grille est complete
        if grille_complete:
            self.current_score = 0.0
            self.game_over = True
            return

        # Jeux de l'adverse
        a = random.choice(self.available_actions_ids())
        if self.pi is not None:
            if self.state_id() in self.pi and random.random() < 0.9 and not self.always_random:
                actions = self.pi[self.state_id()]
                a = max(actions.items(), key=operator.itemgetter(1))[0] if type(actions) is dict else actions
        self.case[a] = 10

        # Toutes les possibiltés de gagner
        ligne0 = self.is_win(case=[0, 1, 2], number_win=30)
        ligne1 = self.is_win(case=[3, 4, 5], number_win=30)
        ligne2 = self.is_win(case=[6, 7, 8], number_win=30)
        colonne0 = self.is_win(case=[0, 3, 6], number_win=30)
        colonne1 = self.is_win(case=[1, 4, 7], number_win=30)
        colonne2 = self.is_win(case=[2, 5, 8], number_win=30)
        diagonal0 = self.is_win(case=[0, 4, 8], number_win=30)
        diagonal1 = self.is_win(case=[2, 4, 6], number_win=30)

        # Verification si l'adversaire gagne
        if ligne0 or ligne1 or ligne2 or colonne0 or colonne1 or colonne2 or diagonal0 or diagonal1:
            self.game_over = True
            self.current_score = 0.0
            return

        # Verification si la grille est complete
        grille_complete = True
        for b in self.case:
            if b == 0:
                grille_complete = False

        # Si la grille est complete : fin du jeu
        if grille_complete:
            self.current_score = 0.0
            self.game_over = True
            return

        self.current_step += 1
        if self.current_step >= self.max_steps:
            self.game_over = True

    def score(self) -> float:
        return self.current_score

    def available_actions_ids(self) -> np.ndarray:
        if self.game_over:
            return np.array([], dtype=np.int)

        n = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int)

        i = 0
        # supression des case qui sont deja joué
        for c in self.case:
            if c != 0:
                n = np.delete(n, i)
            else:
                i += 1
        return n

    def reset(self):
        self.game_over = False
        self.current_step = 0
        self.current_score = 0.0
        self.case = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.always_random = False

    def reset_random(self):
        self.reset()
        self.always_random = False

    def view(self):
        res = '\n' + "-" * 10
        case_str = []
        for b in self.case:
            if b == 1:
                case_str.append('X')
            elif b == 10:
                case_str.append('O')
            else:
                case_str.append('_')

        for index, case in enumerate(case_str):
            if index % 3 == 0:
                res += '\n'
            res += case + ' ' * 3
        res += '\n' + '-' * 10
        print(res)

