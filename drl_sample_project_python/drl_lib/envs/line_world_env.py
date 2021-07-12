import numpy as np
from ..do_not_touch.contracts import MDPEnv


class LineWorld(MDPEnv):
    def __init__(self, cells_nb: int):
        self.cell_nb = cells_nb
        self.__a = np.array([0, 1])  # actions
        self.__r = np.array([-1, 0, 1])  # rewards
        self.__s = np.arrange(self.cell_nb)  # states

    def actions(self) -> np.ndarray:
        return self.__a

    def reward(self) -> np.ndarray:
        return self.__r

    def states(self) -> np.ndarray:
        return self.__s

    def is_state_terminal(self, s: int) -> bool:
        return s == 0 or s == self.cell_nb - 1  # First case or last one

    def transition_probability(self, s: int, a: int, s_p: int, r: float) -> float:
        pass

    def probability(self) -> np.ndarray:
        p = np.zeros((len(self.__s), len(self.__r), len(self.__s), len(self.__a)))  # p(s', r | s, a)

        for s in range(2, self.cell_nb - 1):
            p[s, 1, s - 1, 0] = 1.0

        for s in range(1, self.cell_nb - 2):
            p[s, 1, s + 1, 1] = 1.0

        p[1, 0, 0, 0] = 1.0
        p[self.cell_nb - 2, 2, self.cell_nb - 1, 1] = 1.0

        return p
