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
        self.net = nn.Sequential(
            nn.Linear(state_dimension, self._internal_dimension),
            nn.Tanh(),
            nn.Linear(self._internal_dimension, action_dimension)
        )

        def init_weights(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform_(m.weight)

        self.net.apply(init_weights)


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

    def epoch_update(self, trajectories : List[experience.Trajectory]):

        trainer = torch.optim.Adam(self.policy.net.parameters(), lr=0.01)

        batch_states = []
        batch_actions = []
        batch_returns = []

        for trajectory in trajectories:
            reward_to_go = trajectory.reward
            for state, action, reward in trajectory.state_action_reward():
                batch_states.append(state)
                batch_actions.append(action)
                batch_returns.append(reward_to_go)
                reward_to_go -= reward

        trainer.zero_grad()
        states_tensor = torch.as_tensor(batch_states, dtype=torch.float32)
        actions_tensor = torch.as_tensor(batch_actions, dtype=torch.int32)
        returns_tensor = torch.as_tensor(batch_returns, dtype=torch.float32)

        probabilities = self.policy.net(states_tensor)
        cats = torch.distributions.Categorical(logits=probabilities)
        logp = cats.log_prob(actions_tensor)
        loss = -(logp * returns_tensor).mean()

        loss.backward()
        trainer.step()


env = gym.make('CartPole-v0')
# Action space: Discrete(2)
# State space: Box(4, float32)

total_epochs = 50
total_episodes = 100

epsilon_decay = lambda epoch : 1 / (epoch + 1)

policy = NNPolicy(env.observation_space.shape[0], env.action_space.n)
policy_sampler = CategoricalPolicySampler()
policy_updater = PolicyGradientUpdater(policy)

trainer = training.Trainer()

trainer.run(total_epochs, total_episodes, policy, policy_sampler, policy_updater, env)

total_reward = 0.
for episode in range(total_episodes):
    trajectory = trainer.run_trajectory(policy, policy_sampler, policy_updater, env)
    total_reward += trajectory.reward

print('Final total reward per episode:', total_reward / total_episodes)

env.close()