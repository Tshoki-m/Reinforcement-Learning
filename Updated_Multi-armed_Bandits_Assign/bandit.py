"""
bandit.py

Implements a k-armed bandit environment.

Each arm has:
    - A true mean reward sampled from N(0, 3)
    - Rewards sampled from N(true_mean, 1)

This environment is used by ε-greedy, Optimistic Greedy, and UCB agents.
"""

import numpy as np


class Bandit:
    """
    A k-armed bandit environment.
    """

    def __init__(self, k=10):
        """
        Parameters
        ----------
        k : int
            Number of arms.
        """
        self.k = k

        # True action values (means)
        self.true_values = np.random.normal(
            loc=0.0,
            scale=np.sqrt(3),
            size=k
        )

        # Best possible arm
        self.optimal_arm = np.argmax(self.true_values)

    def pull(self, action):
        """
        Pull an arm and receive a reward.

        Parameters
        ----------
        action : int
            Index of the arm to pull.

        Returns
        -------
        float
            Reward sampled from N(true_value, 1).
        """
        reward = np.random.normal(
            loc=self.true_values[action],
            scale=1.0
        )

        return reward

    def reset(self):
        """
        Generate a completely new bandit.

        This creates new true action values and a new optimal arm.
        """
        self.true_values = np.random.normal(
            loc=0.0,
            scale=np.sqrt(3),
            size=self.k
        )

        self.optimal_arm = np.argmax(self.true_values)

#Testing script Functionality

"""if __name__ == "__main__":
    bandit = Bandit(k=10)

    print("True action values:")
    print(bandit.true_values)

    print("\nOptimal arm:", bandit.optimal_arm)

    print("\nSample rewards from the optimal arm:")
    for _ in range(5):
        print(bandit.pull(bandit.optimal_arm))"""