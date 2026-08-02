import numpy as np

"""
optimistic_greedy.py

Greedy agent with optimistic initialisation.

- Initialises all action-value estimates to Q1 (optimistic).
- Always acts greedily (ε = 0), but early exploration is forced
  because every action looks better than it likely is.
- Updates estimates using incremental sample averages.
"""

import numpy as np


class OptimisticGreedyAgent:
    """
    Greedy agent with optimistic initialisation.

    Parameters
    ----------
    k : int
        Number of arms.
    Q1 : float
        Optimistic initial value for all action-value estimates.
    """

    def __init__(self, num_actions=10, initial_value=5.0):
        self.k = num_actions
        self.Q1 = initial_value

        # Action-value estimates — start optimistically high
        self.Q = np.full(self.k, self.Q1, dtype=float)

        # Count of how many times each action has been taken
        self.N = np.zeros(self.k, dtype=int)

    def select_action(self):
        """
        Select the greedy action (argmax Q), breaking ties randomly.

        Returns
        -------
        int
            Index of the chosen action.
        """
        # Find all actions with the maximum Q-value
        max_q = np.max(self.Q)
        best_actions = np.where(self.Q == max_q)[0]

        # Break ties randomly — CRITICAL at the start when all Q = Q1
        action = np.random.choice(best_actions)
        return action

    def update(self, action, reward):
        """
        Update the action-value estimate using incremental sample average.

        Parameters
        ----------
        action : int
            The action that was taken.
        reward : float
            The reward received.
        """
        self.N[action] += 1

        # Incremental update: Q_new = Q_old + (1/N) * (reward - Q_old)
        step_size = 1.0 / self.N[action]
        self.Q[action] += step_size * (reward - self.Q[action])

    def reset(self):
        """
        Reset the agent for a new run (new bandit task).
        """
        self.Q = np.full(self.k, self.Q1, dtype=float)
        self.N = np.zeros(self.k, dtype=int)
