# Group Members
# Margaret Molefe : 456052
# Sinethemba Nani : 1607717
# Cassia Rene Rieckhoff : 1396207


import numpy as np
from environments.gridworld import GridworldEnv
import timeit
import matplotlib.pyplot as plt


def policy_evaluation(env, policy, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.

    Args:
        env: OpenAI environment.
            env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.observation_space.n is a number of states in the environment.
            env.action_space.n is a number of actions in the environment.
        policy: [S, A] shaped matrix representing the policy.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        Vector of length env.observation_space.n representing the value function.
    """

    num_states = env.observation_space.n
    num_actions = env.action_space.n

    V = np.zeros(num_states)

    while True:
        biggest_change = 0

        for state in range(num_states):
            old_value = V[state]
            new_value = 0

            for action in range(num_actions):
                probability_of_action = policy[state][action]

                for outcome in env.P[state][action]:
                    transition_prob, next_state, reward, done = outcome
                    expected_value_of_this_outcome = reward + discount_factor * V[next_state]
                    new_value += probability_of_action * transition_prob * expected_value_of_this_outcome

            V[state] = new_value

            change = abs(old_value - new_value)

            if change > biggest_change:
                biggest_change = change

        if biggest_change < theta:
            break

    return V
#   raise NotImplementedError


def policy_iteration(env, policy_evaluation_fn=policy_evaluation, discount_factor=1.0):
    """
    Iteratively evaluates and improves a policy until an optimal policy is found.

    Args:
        env: The OpenAI environment.
        policy_evaluation_fn: Policy Evaluation function that takes 3 arguments:
            env, policy, discount_factor.
        discount_factor: gamma discount factor.

    Returns:
        A tuple (policy, V).
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
    """

    def one_step_lookahead(state, V):
        """
        Helper function to calculate the value for all action in a given state.
        """

        action_values = np.zeros(env.action_space.n)

        for action in range(env.action_space.n):

            for probability, next_state, reward, done in env.P[state][action]:

                if done:
                    action_values[action] += probability * reward
                else:
                    action_values[action] += probability * (
                        reward + discount_factor * V[next_state]
                    )

        return action_values

    # Start with a random/uniform policy
    policy = np.ones(
        [env.observation_space.n, env.action_space.n]
    ) / env.action_space.n

    while True:

        # Policy Evaluation
        V = policy_evaluation_fn(
            env,
            policy,
            discount_factor
        )

        # Policy Improvement
        policy_stable = True

        for state in range(env.observation_space.n):

            # Current action according to the policy
            old_action = np.argmax(policy[state])

            # Calculate q(s,a) for every action
            action_values = one_step_lookahead(state, V)

            # Select the best action
            best_action = np.argmax(action_values)

            # Check whether the policy changed
            if old_action != best_action:
                policy_stable = False

            # Update policy to a deterministic policy
            policy[state] = np.zeros(env.action_space.n)
            policy[state, best_action] = 1.0

        # Stop when policy no longer changes
        if policy_stable:
            return policy, V


def value_iteration(env, theta=0.0001, discount_factor=1.0):
    """
    Value Iteration Algorithm.

    Args:
        env: OpenAI environment.
            env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.observation_space.n is a number of states in the environment.
            env.action_space.n is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.
    """

    def one_step_lookahead(state, V):
        """
        Helper function to calculate the value for all action in a given state.

        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.observation_space.n

        Returns:
            A vector of length env.action_space.n containing the expected value of each action.
        """

        A = np.zeros(env.action_space.n)

        for a in range(env.action_space.n):
            for prob, next_state, reward, done in env.P[state][a]:
                A[a] += prob * (
                    reward + discount_factor * V[next_state]
                )

        return A

    V = np.zeros(env.observation_space.n)

    while True:
        delta = 0

        for s in range(env.observation_space.n):
            v = V[s]

            A = one_step_lookahead(s, V)
            V[s] = np.max(A)

            delta = max(delta, abs(v - V[s]))

        if delta < theta:
            break

    policy = np.zeros(
        [env.observation_space.n, env.action_space.n]
    )

    for s in range(env.observation_space.n):
        A = one_step_lookahead(s, V)
        best_action = np.argmax(A)
        policy[s, best_action] = 1.0

    return policy, V


def main():
    # Create Gridworld environment with size of 5 by 5, with the goal at state 24. Reward for getting to goal state is 0, and each step reward is -1
    env = GridworldEnv(
        shape=[5, 5],
        terminal_states=[24],
        terminal_reward=0,
        step_reward=-1
    )

    state = env.reset()
    print("")
    env.render()
    print("")

    # TODO: generate random policy
    trajectory = []
    state = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()
        trajectory.append((state, action))
        next_state, reward, done, info = env.step(action)
        state = next_state

    visited_actions = dict(trajectory)

    action_letters = {
        0: 'U',
        1: 'R',
        2: 'D',
        3: 'L'
    }

    terminal_state = [24]
    symbols_for_states = []

    for s in range(env.observation_space.n):
        if s in terminal_state:
            symbols_for_states.append('X')

        elif s in visited_actions:
            action = visited_actions[s]
            symbols_for_states.append(action_letters[action])

        else:
            symbols_for_states.append('o')

    grid = np.array(symbols_for_states).reshape(env.shape)

    print("Trajectory (random policy):")

    for row in grid:
        print(' '.join(row))

    print("")
    ########################################################################################################

    print("*" * 5 + " Policy evaluation " + "*" * 5)
    print("")

    # TODO: evaluate random policy
    # each state has eq probability of each act
    random_policy = np.ones(
        [env.observation_space.n, env.action_space.n]
    ) / env.action_space.n

    v = policy_evaluation(
        env,
        random_policy,
        discount_factor=1.0
    )

    # TODO: print state value for each state, as grid shape
    print("Value function for uniform random policy:")
    print(v.reshape(env.shape))
    print("")

    # Test: Make sure the evaluated policy is what we expected
    expected_v = np.array([
        -106.81, -104.81, -101.37, -97.62, -95.07,
        -104.81, -102.25, -97.69, -92.40, -88.52,
        -101.37, -97.69, -90.74, -81.78, -74.10,
        -97.62, -92.40, -81.78, -65.89, -47.99,
        -95.07, -88.52, -74.10, -47.99, 0.0
    ])

    np.testing.assert_array_almost_equal(
        v,
        expected_v,
        decimal=2
    )

    print("*" * 5 + " Policy iteration " + "*" * 5)
    print("")

    # TODO: use policy improvement to compute optimal policy and state values
    policy, v = policy_iteration(env, policy_evaluation)  # call policy_iteration

    # TODO Print out best action for each state in grid shape

    # TODO: print state value for each state, as grid shape

    # Test: Make sure the value function is what we expected
    expected_v = np.array([
        -8., -7., -6., -5., -4.,
        -7., -6., -5., -4., -3.,
        -6., -5., -4., -3., -2.,
        -5., -4., -3., -2., -1.,
        -4., -3., -2., -1., 0.
    ])

    np.testing.assert_array_almost_equal(
        v,
        expected_v,
        decimal=1
    )

    #######################################################################################3

    print("*" * 5 + " Value iteration " + "*" * 5)
    print("")

    policy, v = value_iteration(env)

    # Print best action for each state in grid shape
    action_names = {
        0: '↑',
        1: '→',
        2: '↓',
        3: '←'
    }

    policy_grid = np.reshape(
        np.argmax(policy, axis=1),
        env.shape
    )

    print("Optimal Policy:")

    for row in policy_grid:
        print(' '.join([action_names[a] for a in row]))

    print("")

    # Print state value for each state, as grid shape
    v_grid = np.reshape(v, env.shape)

    print("Optimal Value Function:")
    print(v_grid)
    print("")

    # Test: Make sure the value function is what we expected
    expected_v = np.array([
        -8., -7., -6., -5., -4.,
        -7., -6., -5., -4., -3.,
        -6., -5., -4., -3., -2.,
        -5., -4., -3., -2., -1.,
        -4., -3., -2., -1., 0.
    ])

    np.testing.assert_array_almost_equal(
        v,
        expected_v,
        decimal=1
    )

    def plot_runtime_comparison(env):
        discount_rates = np.logspace(-0.2, 0, num=30)
        pi_times = []
        vi_times = []
        n_runs = 10

        for gamma in discount_rates:
            # Time Policy Iteration
            timer_pi = timeit.Timer(
                lambda: policy_iteration(env, policy_evaluation, gamma)
            )

            pi_time = timer_pi.timeit(number=n_runs) / n_runs
            pi_times.append(pi_time)

            # Time Value Iteration
            timer_vi = timeit.Timer(
                lambda: value_iteration(env, discount_factor=gamma)
            )

            vi_time = timer_vi.timeit(number=n_runs) / n_runs
            vi_times.append(vi_time)

        plt.figure(figsize=(10, 6))
        plt.plot(
            discount_rates,
            pi_times,
            marker='o',
            label='Policy Iteration'
        )

        plt.plot(
            discount_rates,
            vi_times,
            marker='s',
            label='Value Iteration'
        )

        plt.xlabel('Discount Rate (γ)')
        plt.ylabel('Average Time (seconds)')

        plt.title(
            f'Policy Iteration vs Value Iteration Runtime '
            f'(avg over {n_runs} runs)'
        )

        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('runtime_comparison.png', dpi=150)
        plt.show()

        print("Plot saved to runtime_comparison.png")

    print("*" * 5 + " Exercise 4.1.2: Runtime Comparison " + "*" * 5)
    plot_runtime_comparison(env)


if __name__ == "__main__":
    main()