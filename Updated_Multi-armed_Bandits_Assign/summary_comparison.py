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



def evaluate_epsilon(epsilon):

    average_reward = 0

    for run in range(NUM_RUNS):

        bandit = Bandit(NUM_ARMS)

        agent = EpsilonGreedyAgent(
            num_actions=NUM_ARMS,
            epsilon=epsilon
        )

        total_reward = 0

        for step in range(NUM_STEPS):

            action = agent.select_action()

            reward = bandit.pull(action)

            agent.update(action, reward)

            total_reward += reward

        average_reward += total_reward / NUM_STEPS

    return average_reward / NUM_RUNS

def evaluate_optimistic(initial_value):

    average_reward = 0

    for run in range(NUM_RUNS):

        bandit = Bandit(NUM_ARMS)

        agent = OptimisticGreedyAgent(
            num_actions=NUM_ARMS,
            initial_value=initial_value
        )

        total_reward = 0

        for step in range(NUM_STEPS):

            action = agent.select_action()

            reward = bandit.pull(action)

            agent.update(action, reward)

            total_reward += reward

        average_reward += total_reward / NUM_STEPS

    return average_reward / NUM_RUNS

def evaluate_ucb(c):

    average_reward = 0

    for run in range(NUM_RUNS):

        bandit = Bandit(NUM_ARMS)

        agent = UCBAgent(
            num_actions=NUM_ARMS,
            c=c
        )

        total_reward = 0

        for step in range(NUM_STEPS):

            action = agent.select_action()

            reward = bandit.pull(action)

            agent.update(action, reward)

            total_reward += reward

        average_reward += total_reward / NUM_STEPS

    return average_reward / NUM_RUNS


def plot_summary_comparison():

    epsilon_values = [1/128,1/64,1/32,1/16,1/8,1/4]
    q_values = [1/4,1/2,1,2,4]
    c_values = [1/16,1/8,1/4,1/2,1,2,4]

    epsilon_rewards = []
    optimistic_rewards = []
    ucb_rewards = []

    for e in epsilon_values:
        epsilon_rewards.append(evaluate_epsilon(e))

    for q in q_values:
        optimistic_rewards.append(evaluate_optimistic(q))

    for c in c_values:
        ucb_rewards.append(evaluate_ucb(c))

    plt.figure(figsize=(10,6))

    plt.plot(
        epsilon_values,
        epsilon_rewards,
        marker='o',
        linewidth=2,
        label='ε-greedy'
    )

    plt.plot(
        c_values,
        ucb_rewards,
        marker='^',
        linewidth=2,
        label='UCB'
    )

    plt.plot(
        q_values,
        optimistic_rewards,
        marker='s',
        linewidth=2,
        label='Optimistic Greedy'
    )

    plt.xscale('log', base=2)
    plt.xticks(
    [1/128,1/64,1/32,1/16,1/8,1/4,1/2,1,2,4],
    ["1/128","1/64","1/32","1/16","1/8","1/4","1/2","1","2","4"]
)

    plt.xlabel(r'$\epsilon$ / $c$ / $Q_1$')

    plt.ylabel("Average Reward over First 1000 Steps")

    plt.title("Summary Comparison of Algorithms")

    plt.grid(True)

    plt.legend()

    plt.show()


if __name__ == "__main__":

    print("Running summary comparison...")

    plot_summary_comparison()

    print("Finished.")