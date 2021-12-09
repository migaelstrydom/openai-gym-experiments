import gym
import policy
import trainer
import episode
import epsilongreedy

ALPHA = 1e-1
GAMMA = 1.0


class SarsaUpdater(policy.PolicyUpdater):

    policy : policy.Policy

    def __init__(self, policy : policy.Policy):
        self.policy = policy

    # No trajectory_update

    def step_update(self, step : episode.Step, next_state : int, next_action : int):
        e = self.policy.experience
        state, action, reward = step.state, step.action, step.reward
        delta = reward + GAMMA * e.Q[next_state, next_action] - e.Q[state, action]
        e.Q[state, action] += ALPHA * delta


env = gym.make('FrozenLake-v1')
# Action space: Discrete(4)
# State space: Discrete(16)

total_epochs = 10
total_episodes = 100

epsilon_decay = lambda epoch : 1 / (epoch + 1)

policy = epsilongreedy.EpsilonGreedyPolicy(env.action_space, epsilon_decay)
policy_updater = SarsaUpdater(policy)

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