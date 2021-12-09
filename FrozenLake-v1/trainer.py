import episode
import policy


def run_trajectory(policy : policy.Policy,
                   policy_updater : policy.PolicyUpdater,
                   environment) -> episode.Trajectory:

    done = False
    steps = []
    state = -1
    action = policy.action(state)

    environment.reset()

    while not done:

        # Take the step
        next_state, reward, done, _ = environment.step(action)

        # Choose the next step based on the current policy
        next_action = policy.action(next_state)

        # Record
        step = episode.Step(action=action, state=state, reward=reward)
        steps.append(step)

        policy_updater.step_update(step, next_state, next_action)

        # Prepare for next iteration
        state = next_state
        action = next_action

    trajectory = episode.Trajectory(steps)
    policy_updater.trajectory_update(trajectory)

    return trajectory


def run(total_epochs : int,
        total_episodes : int,
        policy : policy.Policy,
        policy_updater : policy.PolicyUpdater,
        environment) -> int:

    total_successes = 0

    for epoch in range(total_epochs):

        successes = 0

        policy.update_epoch(epoch)

        for episode in range(total_episodes):

            trajectory = run_trajectory(policy, policy_updater, environment)

            successes += trajectory.reward > 0.0

        total_successes += successes

        print('Successes:', successes)

    return total_successes