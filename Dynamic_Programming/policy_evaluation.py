import numpy as np

from action import Action
from grid import Grid


ACTIONS = [
    Action.UP,
    Action.DOWN,
    Action.LEFT,
    Action.RIGHT
]


def get_next_state(grid, state, action):
    """
    Determine the state reached by taking an action
    from a given state.

    This does not modify the actual agent in the environment.
    """

    row, col = state

    next_state = np.array([row, col])

    if action == Action.UP:
        next_state[0] -= 1

    elif action == Action.DOWN:
        next_state[0] += 1

    elif action == Action.LEFT:
        next_state[1] -= 1

    elif action == Action.RIGHT:
        next_state[1] += 1

    # Boundary condition:
    # if the action would leave the grid,
    # remain in the same state.
    if not grid.is_valid(next_state):
        next_state = np.array([row, col])

    return tuple(next_state)


def policy_evaluation_in_place(gamma=0.9, theta=0.01):
    """
    In-place iterative policy evaluation.

    New values are immediately written into V.
    """

    grid = Grid()

    V = np.zeros((Grid.SIZE, Grid.SIZE))

    iterations = 0

    while True:

        delta = 0

        for row in range(Grid.SIZE):
            for col in range(Grid.SIZE):

                state = (row, col)

                # Terminal state must remain zero.
                if state == tuple(grid.goal_pos):
                    continue

                old_value = V[row, col]

                value = 0

                # Uniform random policy:
                # pi(a|s) = 1/4
                for action in ACTIONS:

                    next_state = get_next_state(
                        grid,
                        state,
                        action
                    )

                    next_row, next_col = next_state

                    reward = -1

                    # Terminal state's future value is zero.
                    if next_state == tuple(grid.goal_pos):
                        future_value = 0
                    else:
                        future_value = V[next_row, next_col]

                    value += 0.25 * (
                        reward + gamma * future_value
                    )

                # IN-PLACE UPDATE
                V[row, col] = value

                delta = max(
                    delta,
                    abs(old_value - value)
                )

        iterations += 1

        if delta < theta:
            break

    # Make absolutely sure terminal state is zero.
    V[0, 0] = 0

    return V, iterations


def policy_evaluation_two_array(gamma=0.9, theta=0.01):
    """
    Two-array iterative policy evaluation.

    V_new is calculated entirely from V.
    V is only replaced after all states
    have been evaluated.
    """

    grid = Grid()

    V = np.zeros((Grid.SIZE, Grid.SIZE))

    iterations = 0

    while True:

        delta = 0

        # Create a separate array for the new values.
        V_new = np.zeros_like(V)

        for row in range(Grid.SIZE):
            for col in range(Grid.SIZE):

                state = (row, col)

                # Terminal state remains zero.
                if state == tuple(grid.goal_pos):
                    V_new[row, col] = 0
                    continue

                value = 0

                # Uniform random policy.
                for action in ACTIONS:

                    next_state = get_next_state(
                        grid,
                        state,
                        action
                    )

                    next_row, next_col = next_state

                    reward = -1

                    if next_state == tuple(grid.goal_pos):
                        future_value = 0
                    else:
                        future_value = V[next_row, next_col]

                    value += 0.25 * (
                        reward + gamma * future_value
                    )

                V_new[row, col] = value

                delta = max(
                    delta,
                    abs(V[row, col] - value)
                )

        # UPDATE ONLY AFTER ALL STATES HAVE BEEN PROCESSED
        V = V_new

        iterations += 1

        if delta < theta:
            break

    V[0, 0] = 0

    return V, iterations