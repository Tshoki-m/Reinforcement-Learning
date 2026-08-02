import numpy as np

"""
epsilon_greedy.py

Epsilon-Greedy agent.

- Selects a random action with probability ε (exploration).
- Otherwise selects the action with the highest estimated value (exploitation).
- Updates action-value estimates using incremental sample averages.
"""

import numpy as np


class EpsilonGreedyAgent:
    """
    Epsilon-Greedy agent.

    Parameters
    ----------
    num_actions : int
        Number of bandit arms.
    epsilon : float
        Probability of selecting a random action.
    """

    def __init__(self, num_actions=10, epsilon=0.1):
        self.num_actions = num_actions
        self.epsilon = epsilon

        # Estimated action values
        self.Q = np.zeros(num_actions, dtype=float)

        # Number of times each action has been selected
        self.N = np.zeros(num_actions, dtype=int)

    def select_action(self):
        """
        Select an action using the ε-greedy policy.

        Returns
        -------
        int
            Index of the selected action.
        """

        # Exploration
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)

        # Exploitation (break ties randomly)
        max_q = np.max(self.Q)
        best_actions = np.where(self.Q == max_q)[0]
        return np.random.choice(best_actions)

    def update(self, action, reward):
        """
        Update the estimated action value using the
        incremental sample-average method.

        Parameters
        ----------
        action : int
            Selected action.
        reward : float
            Reward received from the bandit.
        """

        self.N[action] += 1

        step_size = 1.0 / self.N[action]

        self.Q[action] += step_size * (
            reward - self.Q[action]
        )

    def reset(self):
        """
        Reset the agent for a new bandit problem.
        """

        self.Q = np.zeros(self.num_actions, dtype=float)
        self.N = np.zeros(self.num_actions, dtype=int)