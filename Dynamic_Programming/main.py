import numpy as np
import matplotlib.pyplot as plt

from policy_evaluation import (
    policy_evaluation_in_place,
    policy_evaluation_two_array
)


THETA = 0.01


def plot_value_function(V, gamma):
    """
    Plot the value function as a 2D heatmap.
    """

    fig, ax = plt.subplots(figsize=(7, 6))

    heatmap = ax.imshow(V, cmap="viridis")

    ax.set_title(
        f"Value Function - In-Place Policy Evaluation "
        f"(γ = {gamma})"
    )

    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    ax.set_xticks(range(4))
    ax.set_yticks(range(4))

    # Display numerical values inside each cell
    for row in range(4):
        for col in range(4):
            ax.text(
                col,
                row,
                f"{V[row, col]:.2f}",
                ha="center",
                va="center",
                color="white"
            )

    # Mark the terminal state
    ax.text(
        0,
        0,
        f"{V[0, 0]:.2f}\nGoal",
        ha="center",
        va="center",
        color="white",
        fontweight="bold"
    )

    fig.colorbar(
        heatmap,
        ax=ax,
        label="State Value V(s)"
    )

    plt.tight_layout()
    plt.show()


def plot_convergence(discount_rates, in_place_iterations,
                     two_array_iterations):
    """
    Plot number of iterations to convergence
    against the discount rate for both algorithms.
    """

    plt.figure(figsize=(9, 6))

    plt.plot(
        discount_rates,
        in_place_iterations,
        marker="o",
        label="In-place"
    )

    plt.plot(
        discount_rates,
        two_array_iterations,
        marker="s",
        label="Two-array"
    )

    plt.xlabel("Discount rate γ")
    plt.ylabel("Iterations to convergence")

    plt.title(
        "Policy Evaluation Convergence "
        f"(θ = {THETA})"
    )

    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


def main():

    # =================================================
    # 1. Value function heatmap for gamma = 1
    # =================================================

    gamma = 1.0

    V_in_place, iterations_in_place = (
        policy_evaluation_in_place(
            gamma=gamma,
            theta=THETA
        )
    )

    print("=" * 60)
    print("VALUE FUNCTION FOR γ = 1")
    print("=" * 60)

    print(
        f"In-place iterations: {iterations_in_place}"
    )

    print("\nValue function:")
    print(V_in_place)

    # Plot heatmap
    plot_value_function(
        V_in_place,
        gamma
    )

    # =================================================
    # 2. Policy evaluation for different gamma values
    # =================================================

    discount_rates = np.logspace(
        -0.2,
        0,
        num=20
    )

    in_place_iterations = []
    two_array_iterations = []

    print("\n")
    print("=" * 60)
    print("CONVERGENCE FOR DIFFERENT DISCOUNT RATES")
    print("=" * 60)

    print(
        f"{'Gamma':<12}"
        f"{'In-place':<15}"
        f"{'Two-array':<15}"
    )

    print("-" * 42)

    for gamma in discount_rates:

        # In-place
        _, iterations_in_place = (
            policy_evaluation_in_place(
                gamma=gamma,
                theta=THETA
            )
        )

        # Two-array
        _, iterations_two_array = (
            policy_evaluation_two_array(
                gamma=gamma,
                theta=THETA
            )
        )

        in_place_iterations.append(
            iterations_in_place
        )

        two_array_iterations.append(
            iterations_two_array
        )

        print(
            f"{gamma:<12.4f}"
            f"{iterations_in_place:<15}"
            f"{iterations_two_array:<15}"
        )

    # =================================================
    # Plot convergence results
    # =================================================

    plot_convergence(
        discount_rates,
        in_place_iterations,
        two_array_iterations
    )


if __name__ == "__main__":
    main()