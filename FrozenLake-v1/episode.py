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