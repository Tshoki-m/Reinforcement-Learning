import numpy as np

class e_greedy:
    def __init__(self):
        self.k = k 
        self.epsilon = epsilon
        self.Q = np.zeros(k)
        self.N  =np.zeros(k)

    def action_choice(self):

        if np.random.rand() < self.epsilon:
            return np.random.randint(self.k)

        return np.argmax(self.Q)

   
    def update(self, action, reward):

    self.N[action] += 1

    self.Q[action] += (
        reward - self.Q[action]
    ) / self.N[action]