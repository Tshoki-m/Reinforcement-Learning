"""
experiments.py

Runs MAB experiments and generates plots for:

1. ε-greedy (ε=0.1)
2. Greedy with Optimistic Initialisation (Q1=5)
3. UCB (c=2)

Results averaged over 100 runs.
"""

import numpy as np
import matplotlib.pyplot as plt

from bandit import Bandit
from epsilon_greedy import EpsilonGreedyAgent
from optimistic_greedy import OptimisticGreedyAgent
from ucb import UCBAgent



# Experiment settings

NUM_ARMS = 10
NUM_RUNS = 100
NUM_STEPS = 1000




# MEMBER 1: ε-GREEDY


def run_epsilon_greedy(epsilon=0.1):

    rewards = np.zeros(NUM_STEPS)

    for run in range(NUM_RUNS):

        bandit = Bandit(NUM_ARMS)

        agent = EpsilonGreedyAgent(
            num_actions=NUM_ARMS,
            epsilon=epsilon
        )

        for step in range(NUM_STEPS):

            action = agent.select_action()

            reward = bandit.pull(action)

            agent.update(action, reward)

            rewards[step] += reward


    return rewards / NUM_RUNS




# MEMBER 2: OPTIMISTIC INITIALISATION

def run_optimistic_greedy(initial_value=5):

    rewards = np.zeros(NUM_STEPS)


    for run in range(NUM_RUNS):

        bandit = Bandit(NUM_ARMS)

        agent = OptimisticGreedyAgent(
            num_actions=NUM_ARMS,
            initial_value=initial_value
        )


        for step in range(NUM_STEPS):

            action = agent.select_action()

            reward = bandit.pull(action)

            agent.update(action, reward)

            rewards[step] += reward


    return rewards / NUM_RUNS




# MEMBER 3: UCB

def run_ucb(c=2):

    rewards = np.zeros(NUM_STEPS)


    for run in range(NUM_RUNS):

        bandit = Bandit(NUM_ARMS)

        agent = UCBAgent(
            num_actions=NUM_ARMS,
            c=c
        )


        for step in range(NUM_STEPS):

            action = agent.select_action()

            reward = bandit.pull(action)

            agent.update(action, reward)

            rewards[step] += reward


    return rewards / NUM_RUNS





# REQUIRED PLOT 1
# Fixed parameters
# ε=0.1, Q1=5, c=2

def plot_main_comparison():

    epsilon_rewards = run_epsilon_greedy(0.1)

    optimistic_rewards = run_optimistic_greedy(5)

    ucb_rewards = run_ucb(2)



    plt.figure(figsize=(10,6))

    plt.plot(
        epsilon_rewards,
        label="ε-greedy ε=0.1"
    )

    plt.plot(
        optimistic_rewards,
        label="Optimistic Q1=5"
    )

    plt.plot(
        ucb_rewards,
        label="UCB c=2"
    )


    plt.xlabel("Steps")

    plt.ylabel("Average Reward")

    plt.title(
        "10-Armed Bandit Performance"
    )

    plt.legend()

    plt.grid()

    plt.show()




# REQUIRED PLOT 2
# Hyperparameter comparison

def plot_summary_comparison():
    """
    Plot the average reward over the first 1000 steps for
    ε-greedy, Optimistic Greedy and UCB using different
    hyperparameter values.
    """

    # Hyperparameter values
    epsilon_values = [1/128, 1/64, 1/32, 1/16, 1/8, 1/4]
    q_values = [1/4, 1/2, 1, 2, 4]
    c_values = [1/16, 1/8, 1/4, 1/2, 1, 2, 4]

    epsilon_rewards = []
    optimistic_rewards = []
    ucb_rewards = []

    # ε-greedy
    print("Evaluating ε-greedy...")
    for epsilon in epsilon_values:

        rewards = run_epsilon_greedy(epsilon)

        epsilon_rewards.append(
            np.mean(rewards)
        )

    # Optimistic Greedy
    print("Evaluating Optimistic Greedy...")
    for q in q_values:

        rewards = run_optimistic_greedy(initial_value=q)

        optimistic_rewards.append(
            np.mean(rewards)
        )

    # UCB
    print("Evaluating UCB...")
    for c in c_values:

        rewards = run_ucb(c)

        ucb_rewards.append(
            np.mean(rewards)
        )

    # Plot
    plt.figure(figsize=(10, 6))

    plt.plot(
        epsilon_values,
        epsilon_rewards,
        marker='o',
        linewidth=2,
        label='ε-greedy'
    )

    plt.plot(
        q_values,
        optimistic_rewards,
        marker='s',
        linewidth=2,
        label='Optimistic Greedy'
    )

    plt.plot(
        c_values,
        ucb_rewards,
        marker='^',
        linewidth=2,
        label='UCB'
    )

    # Logarithmic x-axis (base 2)
    plt.xscale("log", base=2)

    plt.xticks(
        [1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4],
        ["1/128", "1/64", "1/32", "1/16",
         "1/8", "1/4", "1/2", "1", "2", "4"]
    )

    plt.xlabel(r'Hyperparameter ($\epsilon$, $Q_1$, $c$)')
    plt.ylabel("Average Reward over First 1000 Steps")
    plt.title("Summary Comparison of Multi-Armed Bandit Algorithms")

    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    # Save figure
    plt.savefig("summary_comparison.png", dpi=300)

    plt.show()


# MAIN

if __name__ == "__main__":


    print("Running experiments...")

    plot_main_comparison()
    plot_summary_comparison()

    print("Finished.")