import numpy as np

class UCBAgent:
    """
    Upper Confidence Bound (UCB) Agent for the k-armed bandit problem.
    """

    def __init__(self, num_actions=10, c=2):
        
        """
        Parameters
        ----------
        num_actions : int
            Number of bandit arms.

        c : float
            Exploration parameter.
        """

        self.num_actions = num_actions
        self.c = c

        # Estimated action values Q(a)
        self.Q = np.zeros(num_actions)

        # Number of times each action has been selected N(a)
        self.N = np.zeros(num_actions)

        # Current time step
        self.t = 1


    def select_action(self):

        """
        Select an action using the Upper Confidence Bound (UCB) rule.
        """

        # Ensure every arm is selected once
        for action in range(self.num_actions):
            if self.N[action] == 0:
                return action

        # Compute UCB value for each arm
        ucb_values = self.Q + self.c * np.sqrt(np.log(self.t) / self.N)

        # Select the arm with the highest UCB value
        return np.argmax(ucb_values)
    

    def update(self, action, reward):
        """
        Update the estimated value of the selected action.

        Parameters
        ----------
        action : int
            The selected arm.

        reward : float
            Reward received from the bandit.
        """

        # Increment time step
        self.t += 1

        # Increment action count
        self.N[action] += 1

        # Update estimated action value using incremental average
        self.Q[action] += (
            reward - self.Q[action]
        ) / self.N[action]