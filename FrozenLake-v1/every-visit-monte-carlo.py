import gym
import random
from collections import namedtuple, defaultdict
from typing import List, Tuple
import policy
import trainer
import episode
import epsilongreedy


class MonteCarloPolicyUpdater(policy.PolicyUpdater):

    policy : policy.Policy

    def __init__(self, policy : policy.Policy):
        self.policy = policy

    def trajectory_update(self, trajectory : episode.Trajectory):
        # The following assignment allows for shorter code.
        e = self.policy.experience

        for state, action in trajectory.state_action():
            e.N[state, action] += 1
            e.S[state, action] += trajectory.reward

        for state, action in trajectory.state_action():
            e.Q[state, action] = e.S[state, action] / e.N[state, action]

    # No step_update


env = gym.make('FrozenLake-v1')
# Action space: Discrete(4)
# State space: Discrete(16)

total_epochs = 10
total_episodes = 1000

policy = epsilongreedy.EpsilonGreedyPolicy(env.action_space)
policy_updater = MonteCarloPolicyUpdater(policy)

trainer.run(total_epochs, total_episodes, policy, policy_updater, env)

greedy_policy = policy.greedy

successes = 0
for episode in range(total_episodes):
    trajectory = trainer.run_trajectory(greedy_policy, policy_updater, env)
    successes += trajectory.reward > 0.0

print('Successes on greedy policy:', successes)

#for state, action in Q:
#    print(state, action, ':', Q[state, action])

env.close()