from . import experience
from . import policy
import numpy as np


class Trainer:

    exp_buffer : experience.ExperienceBuffer

    def __init__(self, gamma : float = 0.99, lam : float = 0.95):
        self.exp_buffer = experience.ExperienceBuffer(gamma, lam)

    def run_trajectory(self,
                       policy : policy.Policy,
                       policy_sampler : policy.PolicySampler,
                       policy_updater : policy.PolicyUpdater,
                       environment) -> experience.Trajectory:

        done = False
        steps = []
        # environment.reset() returns an observation
        state = environment.reset()
        action = policy_sampler.action(policy, state)

        while not done:

            # Take the step
            next_state, reward, done, _ = environment.step(action)

            # Choose the next step based on the current policy
            next_action = policy_sampler.action(policy, next_state.copy())

            # Record
            step = experience.Step(action=action, state=state.copy(), reward=reward)
            steps.append(step)

            policy_updater.step_update(step, next_state.copy(), next_action)

            # Prepare for next iteration
            state = next_state
            action = next_action

        trajectory = experience.Trajectory(steps)
        policy_updater.trajectory_update(trajectory)

        return trajectory

    def run(self,
            total_epochs : int,
            total_episodes : int,
            policy : policy.Policy,
            policy_sampler : policy.PolicySampler,
            policy_updater : policy.PolicyUpdater,
            environment):

        for epoch in range(total_epochs):

            total_reward = 0.0
            total_len = 0
            self.exp_buffer.new_epoch()

            policy_sampler.update_epoch(epoch)

            for _ in range(total_episodes):

                trajectory = self.run_trajectory(policy, policy_sampler, policy_updater, environment)

                total_reward += trajectory.reward
                total_len += len(trajectory.steps)

                self.exp_buffer.add(trajectory)

            self.exp_buffer.finish_epoch()
            policy_updater.epoch_update(self.exp_buffer.latest_epoch())

            print('Epoch: %3d\tAverage epoch reward: %3f \tAverage episode len: %3f' % (epoch, total_reward / total_episodes, total_len / total_episodes))

    def display_trajectory(self,
                           policy : policy.Policy,
                           policy_sampler : policy.PolicySampler,
                           environment):
        done = False
        # environment.reset() returns an observation
        state = environment.reset()
        action = policy_sampler.action(policy, state)

        while not done:

            environment.render()

            # Take the step
            next_state, reward, done, _ = environment.step(action)

            # Choose the next step based on the current policy
            next_action = policy_sampler.action(policy, next_state.copy())

            # Prepare for next iteration
            state = next_state
            action = next_action