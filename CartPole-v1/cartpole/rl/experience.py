from collections import namedtuple
from typing import List

# A step in the trajectory
Step = namedtuple('Step', ['action', 'state', 'reward'])


class Trajectory:
    steps : List[Step]
    reward : float

    def __init__(self, steps):
        self.steps = steps
        self.reward = sum([s.reward for s in steps])

    def state_action(self):
        return [(s.state, s.action) for s in self.steps]

    def state_action_reward(self):
        return [(s.state, s.action, s.reward) for s in self.steps]


class ExperienceBuffer:
    trajectories : List[Trajectory] = []

    def add(self, trajectory : Trajectory):
        self.trajectories.append(trajectory)

    def positive_rewards(self):
        return sum([t.reward > 0.0 for t in self.trajectories])
