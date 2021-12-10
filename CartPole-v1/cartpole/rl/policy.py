from abc import ABC
from collections import defaultdict
from . import experience
import numpy as np
from typing import List


# A policy is a distribution over actions given states. $\pi(a | s)$.
class Policy(ABC):

    def actions(self, state : np.array) -> np.array:
        pass


# A way of sampling from a policy, such as greedy or epsilon-greedy.
class PolicySampler(ABC):

    def action(self, pi : Policy, state : np.array) -> int:
        pass

    def update_epoch(self, epoch : int):
        pass


# A PolicyUpdater represents a particular learning algorithm.
class PolicyUpdater(ABC):

    def trajectory_update(self, trajectory : experience.Trajectory):
        pass

    def step_update(self, step : experience.Step, next_state : np.array, next_action : np.array):
        pass

    def epoch_update(self, trajectories : List[experience.Trajectory]):
        pass


# The learned state
class Learned:
    # Number of times the state is seen
    N = defaultdict(lambda: 0)
    # Accumulated reward
    S = defaultdict(lambda: 0.0)
    # Policy
    pi : Policy

    def __init__(self, pi : Policy):
        self.pi = pi