import unittest
from cartpole.rl import policy


class TestPolicy(unittest.TestCase):

    def test_dense_network(self):

        # Execute
        under_test = policy.dense_network([1, 2, 3])

        print(under_test)

        # Verify
        self.assertEqual(under_test[0].in_features, 1)
        self.assertEqual(under_test[0].out_features, 2)
        self.assertEqual(under_test[2].in_features, 2)
        self.assertEqual(under_test[2].out_features, 3)


if __name__ == '__main__':
    unittest.main()