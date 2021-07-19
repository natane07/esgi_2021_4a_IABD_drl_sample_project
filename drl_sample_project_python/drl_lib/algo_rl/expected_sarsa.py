import numpy as np
from tqdm import *
from ..do_not_touch.contracts import SingleAgentEnv
from ..do_not_touch.result_structures import PolicyAndActionValueFunction



def expected_sarsa(
        env: SingleAgentEnv,
        alpha: float,
        epsilon: float,
        gamma: float,
        max_iter: int) -> PolicyAndActionValueFunction:
    pass
