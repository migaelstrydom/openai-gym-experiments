import gym
import numpy as np
import torch
from torch import nn
from typing import List
from rl import policy
from rl import training
from rl import experience

class NNPolicy(policy.Policy):

    _internal_dimension : int = 32
    net : nn.Module

    def __init__(self, state_dimension : int, action_dimension : int):
        self.net = policy.dense_network([state_dimension,
                                         self._internal_dimension,
                                         action_dimension],
                                         activation = nn.Tanh)


class CategoricalPolicySampler(policy.PolicySampler):

    def action(self, pi : policy.Policy, state : np.array) -> int:
        input = torch.as_tensor(state, dtype=torch.float32)
        probabilities = pi.net(input)
        cats = torch.distributions.Categorical(logits=probabilities)
        return cats.sample().item()


class PolicyGradientUpdater(policy.PolicyUpdater):

    policy : policy.Policy

    def __init__(self, policy : policy.Policy):
        self.policy = policy

    def epoch_update(self, epoch_trajectories : experience.EpochTrajectories):

        trainer = torch.optim.Adam(self.policy.net.parameters(), lr=0.01)

        batch_states = []
        batch_actions = []
        batch_returns = []

        for trajectory in epoch_trajectories.trajectories:
            reward_to_go = trajectory.reward
            for state, action, reward in trajectory.state_action_reward():
                batch_states.append(state)
                batch_actions.append(action)
                batch_returns.append(reward_to_go)
                reward_to_go -= reward

        states_tensor = torch.as_tensor(batch_states, dtype=torch.float32)
        actions_tensor = torch.as_tensor(batch_actions, dtype=torch.int32)
        returns_tensor = torch.as_tensor(batch_returns, dtype=torch.float32)

        for _ in range(10):
            trainer.zero_grad()

            probabilities = self.policy.net(states_tensor)
            cats = torch.distributions.Categorical(logits=probabilities)
            logp = cats.log_prob(actions_tensor)
            loss = -(logp * returns_tensor).mean()

            loss.backward()
            trainer.step()


env = gym.make('Acrobot-v1')
# Action space: Discrete(2)
# State space: Box(4, float32)

total_epochs = 10
total_episodes = 100

pi = NNPolicy(env.observation_space.shape[0], env.action_space.n)
policy_sampler = CategoricalPolicySampler()
policy_updater = PolicyGradientUpdater(pi)

trainer = training.Trainer()

trainer.run(total_epochs, total_episodes, pi, policy_sampler, policy_updater, env)

trainer.display_trajectory(pi, policy_sampler, env)

env.close()