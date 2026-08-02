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
             k=NUM_ARMS,
            Q1=initial_value
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

def plot_hyperparameters():


    plt.figure(figsize=(10,6))


    # ε-greedy values

    for epsilon in [0,0.01,0.1,0.2]:

        rewards = run_epsilon_greedy(epsilon)

        plt.plot(
            rewards,
            label=f"ε={epsilon}"
        )


    plt.xlabel("Steps")

    plt.ylabel("Average Reward")

    plt.title(
        "ε-greedy Hyperparameter Comparison"
    )

    plt.legend()

    plt.grid()

    plt.show()



# MAIN

if __name__ == "__main__":


    print("Running experiments...")


    plot_main_comparison()


    plot_hyperparameters()


    print("Finished.")