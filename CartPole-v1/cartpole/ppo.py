import gym
import numpy as np
import scipy.signal
import torch
from torch import nn
from typing import List
from rl import policy
from rl import training
from rl import experience

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class NNPolicy(policy.Policy):

    net : nn.Module

    def __init__(self, state_dimension : int, action_dimension : int):
        self.net = policy.dense_network([state_dimension,
                                         32,
                                         16,
                                         action_dimension],
                                         activation=nn.Tanh)

class NNValue:

    net : nn.Module

    def __init__(self, state_dimension : int):
        self.net = policy.dense_network([state_dimension,
                                         32,
                                         16,
                                         1],
                                         activation=nn.Tanh)


class CategoricalPolicySampler(policy.PolicySampler):

    def action(self, pi : policy.Policy, state : np.array) -> int:
        input = torch.as_tensor(state, dtype=torch.float32)
        probabilities = pi.net(input)
        cats = torch.distributions.Categorical(logits=probabilities)
        return cats.sample().item()


class PPOUpdater(policy.PolicyUpdater):

    policy : policy.Policy
    value : NNValue

    gamma : float
    lam : float
    clamp_ratio : float

    def __init__(self,
                 policy : policy.Policy,
                 value : NNValue,
                 gamma : float = 0.99,
                 lam : float = 0.95,
                 clamp_ratio : float = 0.2):
        self.policy = policy
        self.value = value
        self.gamma = gamma
        self.lam = lam
        self.clamp_ratio = clamp_ratio

    def calculate_advantage(self, rewards : np.array, states : np.array):

        states_tensor = torch.as_tensor(states, dtype=torch.float32)
        values = self.value.net(states_tensor).squeeze().detach().numpy()

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        advantage = np.zeros(len(rewards))
        advantage[:-1] = discount_cumsum(deltas, self.gamma * self.lam)

        return advantage

    def epoch_update(self, epoch_trajectories : experience.EpochTrajectories):
        cat = np.concatenate
        batch_advantage = cat([self.calculate_advantage(t.rewards(), t.states()) for t in epoch_trajectories.trajectories])

        pi_trainer = torch.optim.Adam(self.policy.net.parameters(), lr=3e-4)
        v_trainer = torch.optim.Adam(self.value.net.parameters(), lr=1e-3)

        states_tensor = torch.as_tensor(epoch_trajectories.batch_states, dtype=torch.float32)
        actions_tensor = torch.as_tensor(epoch_trajectories.batch_actions, dtype=torch.int32)
        returns_tensor = torch.as_tensor(epoch_trajectories.batch_returns, dtype=torch.float32)
        steps_tensor = torch.as_tensor(epoch_trajectories.batch_steps, dtype=torch.int32)
        advantage_tensor = torch.as_tensor(batch_advantage, dtype=torch.float32)

        with torch.no_grad():
            pi = self.policy.net(states_tensor)
            cats = torch.distributions.Categorical(logits=pi)
            logp_old = cats.log_prob(actions_tensor)

        for _ in range(80):
            pi_trainer.zero_grad()
            pi = self.policy.net(states_tensor)
            cats = torch.distributions.Categorical(logits=pi)
            logp = cats.log_prob(actions_tensor)

            ratio = torch.exp(logp - logp_old)
            clamped_ratio = torch.clamp(ratio, 1 - self.clamp_ratio, 1 + self.clamp_ratio)

            pi_loss = -torch.min(ratio * advantage_tensor, clamped_ratio * advantage_tensor).mean()
            pi_loss.backward()
            pi_trainer.step()

        for _ in range(80):
            v_trainer.zero_grad()
            values = self.value.net(states_tensor).squeeze()
            v_loss = (((values - returns_tensor))**2).mean()
            v_loss.backward()
            v_trainer.step()




env = gym.make('CartPole-v1')
# Action space: Discrete(2)
# State space: Box(4, float32)

total_epochs = 10
total_episodes = 500

pi = NNPolicy(env.observation_space.shape[0], env.action_space.n)
value_function = NNValue(env.observation_space.shape[0])
policy_sampler = CategoricalPolicySampler()
policy_updater = PPOUpdater(pi, value_function)

trainer = training.Trainer()

trainer.run(total_epochs, total_episodes, pi, policy_sampler, policy_updater, env)

trainer.display_trajectory(pi, policy_sampler, env)

env.close()