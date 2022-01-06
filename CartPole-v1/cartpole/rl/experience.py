from collections import namedtuple
from typing import List
import numpy as np

# A step in the trajectory
Step = namedtuple('Step', ['action', 'state', 'reward'])


class Trajectory:
    steps : List[Step]
    reward : float

    def __init__(self, steps):
        self.steps = steps
        self.reward = sum([s.reward for s in steps])

    def states(self):
        return np.array([s.state for s in self.steps])

    def actions(self):
        return np.array([s.action for s in self.steps])

    def rewards(self):
        return np.array([s.reward for s in self.steps])

    def state_action(self):
        return [(s.state, s.action) for s in self.steps]

    def state_action_reward(self):
        return [(s.state, s.action, s.reward) for s in self.steps]


class EpochTrajectories:
    trajectories : List[Trajectory]
    batch_states : np.array
    batch_actions : np.array
    batch_returns : np.array
    batch_steps : np.array

    def __init__(self):
        self.trajectories = []

    def add(self, trajectory : Trajectory):
        self.trajectories.append(trajectory)

    def calculate_returns(self, rewards : np.array):
        return rewards.sum() - np.concatenate([[0], np.cumsum(rewards)[:-1]])

    def calculate_steps_to_go(self, total_steps):
        return np.array(range(total_steps, 0, -1))

    """
    Called at the end of an epoch, and calculates batch lists of states, actions, returns and steps,
    for all of the trajectories in the epoch.
    """
    def finish(self):
        cat = np.concatenate
        self.batch_states = cat([t.states() for t in self.trajectories])
        self.batch_actions = cat([t.actions() for t in self.trajectories])
        self.batch_returns = cat([self.calculate_returns(t.rewards()) for t in self.trajectories])
        self.batch_steps = cat([self.calculate_steps_to_go(len(t.steps)) for t in self.trajectories])


class ExperienceBuffer:
    epochs : List[EpochTrajectories]
    gamma : float
    lam : float

    def __init__(self, gamma : float = 0.99, lam : float = 0.95):
        self.epochs = []
        self.gamma = gamma
        self.lam = lam

    def add(self, trajectory : Trajectory):
        self.latest_epoch().add(trajectory)

    def new_epoch(self):
        self.epochs.append(EpochTrajectories())

    def finish_epoch(self):
        self.latest_epoch().finish()

    def latest_epoch(self) -> EpochTrajectories:
        return self.epochs[-1]
