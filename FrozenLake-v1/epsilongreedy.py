import gym
import policy
import random
from typing import Callable

class GreedyPolicy(policy.Policy):

    action_space : gym.spaces.space.Space

    def __init__(self, action_space : gym.spaces.space.Space):
        self.action_space = action_space

    def action(self, state : int) -> int:
        score = self.experience.Q[state, 0]
        best_action = 0
        action = 1

        while action < self.action_space.n:
            if score < self.experience.Q[state, action]:
                best_action = action
                score = self.experience.Q[state, action]
            action += 1

        return best_action


class EpsilonGreedyPolicy(policy.Policy):
    EPSILON = 1e-1
    action_space : gym.spaces.space.Space
    greedy : GreedyPolicy
    epsilon_decay : Callable[[int], int]

    def __init__(self, action_space : gym.spaces.space.Space, epsilon_decay : Callable[[int], int] = None):
        self.action_space = action_space
        self.greedy = GreedyPolicy(action_space)
        self.epsilon_decay = epsilon_decay

    def action(self, state : int) -> int:
        if random.random() < self.EPSILON:
            return self.action_space.sample()
        else:
            return self.greedy.action(state)

    def update_epoch(self, epoch : int):
        if self.epsilon_decay is not None:
            self.EPSILON = self.epsilon_decay(epoch)