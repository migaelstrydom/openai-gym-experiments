import unittest
import numpy as np
from cartpole.rl import experience

class TestPolicy(unittest.TestCase):

    def test_experience_buffer_initialization(self):

        # Execute
        under_test = experience.ExperienceBuffer()

        # Verify
        self.assertEqual(len(under_test.epochs), 0)
        self.assertEqual(under_test.gamma, 0.99)
        self.assertEqual(under_test.lam, 0.95)

    def test_experience_buffer_add(self):

        # Setup
        under_test = experience.ExperienceBuffer()

        # Execute
        under_test.new_epoch()
        under_test.add(experience.Trajectory([]))
        under_test.add(experience.Trajectory([]))
        under_test.new_epoch()
        under_test.add(experience.Trajectory([]))
        under_test.add(experience.Trajectory([]))
        under_test.add(experience.Trajectory([]))

        # Verify
        self.assertEqual(len(under_test.epochs), 2)
        self.assertEqual(len(under_test.epochs[0].trajectories), 2)
        self.assertEqual(len(under_test.epochs[1].trajectories), 3)

    def test_calculate_returns(self):

        # Setup
        under_test = experience.EpochTrajectories()

        rewards = np.array([1, 2, 3, 4, 5])

        # Execute
        result = under_test.calculate_returns(rewards)

        # Verify
        expected = np.array([15, 14, 12, 9, 5])

        self.assertTrue(np.array_equal(result, expected))

    def test_calculate_steps_to_go(self):

        # Setup
        under_test = experience.EpochTrajectories()

        # Execute
        result = under_test.calculate_steps_to_go(5)

        # Verify
        expected = np.array([5, 4, 3, 2, 1])

        self.assertTrue(np.array_equal(result, expected))


if __name__ == '__main__':
    unittest.main()