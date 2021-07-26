import numpy as np
from ..do_not_touch.contracts import MDPEnv


class GridWorld(MDPEnv):
    def __init__(self, rows: int, columns: int):
        self.rows = rows
        self.columns = columns
        self.cell_nb = rows * columns
        self.__a = np.array([0, 1, 2, 3])  # actions : 0 -> left, 1 -> right, 2 -> up, 3 -> down
        self.__r = np.array([-1, 0, 1])  # rewards
        self.__s = np.arange(self.cell_nb)  # states
        self.bad_end = 4
        self.good_end = self.cell_nb - 1
        self.p = self.probability()
        self.name_env = "Grid World"

    def actions(self) -> np.ndarray:
        return self.__a

    def rewards(self) -> np.ndarray:
        return self.__r

    def states(self) -> np.ndarray:
        return self.__s

    def is_state_terminal(self, s: int) -> bool:
        return s == self.good_end or s == self.bad_end

    def transition_probability(self, s: int, a: int, s_p: int, r: float) -> float:
        return self.p[s_p, r, s, a]

    def probability(self) -> np.ndarray:
        p = np.zeros((len(self.__s), len(self.__r), len(self.__s), len(self.__a)))  # p(s', r | s, a)
        for row in range(0, self.rows):
            for column in range(0, self.columns - 1):
                # To the right
                s = row * self.columns + column
                if s != self.good_end and s != self.bad_end:
                    if s + 1 == self.good_end:
                        p[s + 1, 2, s, 1] = 1.0
                    elif s + 1 == self.bad_end:
                        p[s + 1, 0, s, 1] = 1.0
                    else:
                        p[s + 1, 1, s, 1] = 1.0

                # To the left
                s = row * self.columns + column + 1
                if s != self.good_end and s != self.bad_end:
                    if s - 1 == self.good_end:
                        p[s - 1, 2, s, 0] = 1.0
                    elif s - 1 == self.bad_end:
                        p[s - 1, 0, s, 0] = 1.0
                    else:
                        p[s - 1, 1, s, 0] = 1.0

        # To the up
        for column in range(0, self.columns):
            for row in range(0, self.rows - 1):
                # To the up
                s = row * self.columns + column
                s_lower = (row + 1) * self.columns + column
                if s_lower != self.good_end and s_lower != self.bad_end:
                    if s == self.good_end:
                        p[s, 2, s_lower, 2] = 1.0
                    elif s == self.bad_end:
                        p[s, 0, s_lower, 2] = 1.0
                    else:
                        p[s, 1, s_lower, 2] = 1.0

                # To the down
                if s != self.good_end and s != self.bad_end:
                    if s == self.good_end:
                        p[s_lower, 2, s, 3] = 1.0
                    elif s == self.bad_end:
                        p[s_lower, 0, s, 3] = 1.0
                    else:
                        p[s_lower, 1, s, 3] = 1.0

        return p
