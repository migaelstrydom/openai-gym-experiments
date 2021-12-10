import unittest
from cartpole.rl import epsilongreedy, policy
import numpy as np


class DummyPolicy(policy.Policy):

    def actions(self, state : np.array) -> np.array:
        return np.array([0.1, 0.0, 0.3, 0.2, 0.1])


class TestEpsilonGreedy(unittest.TestCase):

    def test_greedy_action(self):

        # Set up
        policy = DummyPolicy()
        under_test = epsilongreedy.GreedyPolicySampler()

        # Execute
        result = under_test.action(policy, 0)

        # Verify
        self.assertEqual(result, 2)

if __name__ == '__main__':
    unittest.main()