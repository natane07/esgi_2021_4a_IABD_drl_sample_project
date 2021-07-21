import operator
import random
import numpy as np
from ..do_not_touch.contracts import SingleAgentEnv


class EnvTicTacToeSingleAgent(SingleAgentEnv):
    def __init__(self, max_steps: int, first_player: int = 0, pi=None):
        assert (max_steps > 0)
        self.max_steps = max_steps
        self.is_first_player = first_player  # 0: random, 1: always first, 2: always second
        self.pi = pi
        self.reset()

    def state_id(self) -> int:
        num = ''
        for c in self.case:
            if c == 0:
                num += '1'
            elif c == 1:
                num += '2'
            else:
                num += '3'

        return int(num)

    def is_game_over(self) -> bool:
        return self.game_over

    def act_with_action_id(self, action_id: int):
        assert (not self.game_over)
        assert (action_id in [0, 1, 2, 3, 4, 5, 6, 7, 8])
        self.case[action_id] = 1

        # Toutes les possibiltés de gagner
        ligne0 = self.case[0] + self.case[1] + self.case[2]
        ligne1 = self.case[3] + self.case[4] + self.case[5]
        ligne2 = self.case[6] + self.case[7] + self.case[8]
        colonne0 = self.case[0] + self.case[3] + self.case[6]
        colonne1 = self.case[1] + self.case[4] + self.case[7]
        colonne2 = self.case[2] + self.case[5] + self.case[8]
        diagonal0 = self.case[0] + self.case[4] + self.case[8]
        diagonal1 = self.case[2] + self.case[4] + self.case[6]

        # Verification si la grille est complete
        grille_complete = True
        for b in self.case:
            if b == 0:
                grille_complete = False

        # Verification si il y a une un gagnant
        if ligne0 == 3 or ligne1 == 3 or ligne2 == 3 or colonne0 == 3 or colonne1 == 3 or colonne2 == 3 or diagonal0 == 3 or diagonal1 == 3:
            self.game_over = True
            self.current_score = 1.0
            return

        # Si la grille est complete
        if grille_complete:
            self.current_score = -1.0
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
        ligne0 = self.case[0] + self.case[1] + self.case[2]
        ligne1 = self.case[3] + self.case[4] + self.case[5]
        ligne2 = self.case[6] + self.case[7] + self.case[8]
        colonne0 = self.case[0] + self.case[3] + self.case[6]
        colonne1 = self.case[1] + self.case[4] + self.case[7]
        colonne2 = self.case[2] + self.case[5] + self.case[8]
        diagonal0 = self.case[0] + self.case[4] + self.case[8]
        diagonal1 = self.case[2] + self.case[4] + self.case[6]

        # Verification si l'adversaire gagne
        if ligne0 == 30 or ligne1 == 30 or ligne2 == 30 or colonne0 == 30 or colonne1 == 30 or colonne2 == 30 or diagonal0 == 30 or diagonal1 == 30:
            self.game_over = True
            self.current_score = -1.0
            return

        # Verification si la grille est complete
        grille_complete = True
        for b in self.case:
            if b == 0:
                grille_complete = False

        # Si la grille est complete : fin du jeu
        if grille_complete:
            self.current_score = -1.0
            self.game_over = True
            return

        self.current_step += 1
        if self.current_step >= self.max_steps:
            self.game_over = True

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


    def is_case_valid(self) -> bool:
        # Il y a t-il des case vide
        case_not_empty = False
        for i in range(len(self.case)):
            if self.case[i] == 0:
                case_not_empty = True

        if not case_not_empty:
            return False

        # Toutes les possibiltés de gagner
        ligne0 = self.case[0] + self.case[1] + self.case[2]
        ligne1 = self.case[3] + self.case[4] + self.case[5]
        ligne2 = self.case[6] + self.case[7] + self.case[8]
        colonne0 = self.case[0] + self.case[3] + self.case[6]
        colonne1 = self.case[1] + self.case[4] + self.case[7]
        colonne2 = self.case[2] + self.case[5] + self.case[8]
        diagonal0 = self.case[0] + self.case[4] + self.case[8]
        diagonal1 = self.case[2] + self.case[4] + self.case[6]

        # Verification d'un gagnant
        if ligne0 == 3 or ligne1 == 3 or ligne2 == 3 or colonne0 == 3 or colonne1 == 3 or colonne2 == 3 or diagonal0 == 3 or diagonal1 == 3:
            return False
        if ligne0 == 30 or ligne1 == 30 or ligne2 == 30 or colonne0 == 30 or colonne1 == 30 or colonne2 == 30 or diagonal0 == 30 or diagonal1 == 30:
            return False
        return True

    def reset_random(self):
        self.reset()
        self.always_random = False

