from abc import ABC
from collections import defaultdict
import episode

# The learned state
class Experience:
    # Number of times the state is seen
    N = defaultdict(lambda: 0)
    # Accumulated reward
    S = defaultdict(lambda: 0.0)
    # Action-value function
    Q = defaultdict(lambda: 0.0)
    # Eligibility Trace
    E = defaultdict(lambda: 0.0)


# A policy is a distribution over actions given states. $\pi(a | s)$.
class Policy(ABC):

    experience = Experience()

    def action(self, state : int) -> int:
        pass

    def update_epoch(self, epoch : int):
        pass


# A PolicyUpdater represents a particular learning algorithm.
class PolicyUpdater(ABC):
    def trajectory_update(self, trajectory : episode.Trajectory):
        pass

    def step_update(self, step : episode.Step, next_state : int, next_action : int):
        pass
