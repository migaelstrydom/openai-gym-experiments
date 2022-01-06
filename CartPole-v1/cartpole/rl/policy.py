from abc import ABC
from collections import defaultdict
from . import experience
import numpy as np
from typing import List
from torch import nn


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

    def epoch_update(self, trajectories : experience.EpochTrajectories):
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


def dense_network(layer_sizes : List[int],
                  activation : nn.Module = nn.Tanh,
                  final_activation : nn.Module = nn.Identity) -> nn.Sequential:
    layers = []
    for li in range(len(layer_sizes) - 1):
        layers.append(nn.Linear(layer_sizes[li], layer_sizes[li + 1]))
        if li == len(layer_sizes) - 2:
            layers.append(final_activation())
        else:
            layers.append(activation())

    net = nn.Sequential(*layers)

    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)

    return net