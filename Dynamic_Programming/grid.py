import numpy as np

from action import Action


class Grid:

    SIZE = 4
    STEP_REWARD = -1

    def __init__(self):

        # Create an empty 4x4 grid
        self.grid = np.zeros((self.SIZE, self.SIZE))

        # Goal is in the top-left corner
        self.goal_pos = np.array([0, 0])

        # Start position
        self.start_pos = np.array([3, 3])

        # Agent starts at the start position
        self.agent_pos = self.start_pos.copy()

    def reset(self):
        self.agent_pos = self.start_pos.copy()
        return self.agent_pos.copy()

    def is_terminal(self):
        return np.array_equal(self.agent_pos, self.goal_pos)

    def is_valid(self, position):

        row = position[0]
        col = position[1]

        return (
            0 <= row < self.SIZE
            and 0 <= col < self.SIZE
        )

    def step(self, action):

        new_position = self.agent_pos.copy()

        if action == Action.UP:
            new_position[0] -= 1

        elif action == Action.DOWN:
            new_position[0] += 1

        elif action == Action.LEFT:
            new_position[1] -= 1

        elif action == Action.RIGHT:
            new_position[1] += 1

        # If the move would leave the grid,
        # remain in the current state.
        if self.is_valid(new_position):
            self.agent_pos = new_position

        # Every transition receives -1,
        # including entering the goal.
        reward = self.STEP_REWARD

        # Entering the goal ends the episode.
        done = self.is_terminal()

        return self.agent_pos.copy(), reward, done