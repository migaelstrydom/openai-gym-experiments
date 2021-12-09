import gym
import policy
import trainer
import episode
import epsilongreedy

from collections import defaultdict

ALPHA = 1e-1
GAMMA = 1.0
LAMBDA = 0.1


class SarsaLambdaUpdater(policy.PolicyUpdater):

    policy : policy.Policy

    def __init__(self, policy : policy.Policy):
        self.policy = policy

    # No trajectory_update

    def step_update(self, step : episode.Step, next_state : int, next_action : int):
        e = self.policy.experience
        state, action, reward = step.state, step.action, step.reward

        e.E[state, action] += 1
        delta = reward + GAMMA * e.Q[next_state, next_action] - e.Q[state, action]

        seen = defaultdict(lambda: False)
        for s, a in e.E:
            if not seen[s, a]:
                e.Q[s, a] += ALPHA * delta * e.E[s, a]
                e.E[s, a] *= GAMMA * LAMBDA
            seen[s, a] = True


env = gym.make('FrozenLake-v1')
# Action space: Discrete(4)
# State space: Discrete(16)

total_epochs = 10
total_episodes = 1000

epsilon_decay = lambda epoch : 1 / (epoch + 1)

policy = epsilongreedy.EpsilonGreedyPolicy(env.action_space, epsilon_decay)
policy_updater = SarsaLambdaUpdater(policy)

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