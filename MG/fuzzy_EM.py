import random

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import norm, multivariate_normal
from sklearn.cluster import KMeans

import models.trainable.fuzzy_EM
from utils import utils
from continouos_pomdp_example import (generate_pomdp_data,
                                      STATES)
from fuzzy.fuzzy_model import build_fuzzymodel
from models.trainable.pomdp_EM import PomdpEM

from MG.MG_FM import _simulate_data, build_fuzzy_model

OBS_LIST = [
    "Teff", "Treg", "B", "GC", "SLPB", "LLPC", "IgG", "Complement", "Symptoms", "Inflammation",
]
obs_var_index = {
    "Teff": 0, "Treg": 1, "B": 2, "GC": 3, "SLPB": 4,
    "LLPC": 5, "IgG": 6, "Complement": 7, "Symptoms": 8, "Inflammation": 9
}


def plot_sensitivity_results(results):
    """
    Plot the sensitivity analysis results as a heatmap.

    Args:
        results: Dictionary with keys 'rho_T=X, rho_O=Y' and distance values
    """
    # Extract unique rho values
    rho_T_values = []
    rho_O_values = []

    for key in results.keys():
        parts = key.split(',')
        rho_T = float(parts[0].split('=')[1])
        rho_O = float(parts[1].split('=')[1])

        if rho_T not in rho_T_values:
            rho_T_values.append(rho_T)
        if rho_O not in rho_O_values:
            rho_O_values.append(rho_O)

    # Sort the values
    rho_T_values.sort()
    rho_O_values.sort()

    # Create the heatmap data
    heatmap_data = np.zeros((len(rho_T_values), len(rho_O_values)))

    for i, rho_T in enumerate(rho_T_values):
        for j, rho_O in enumerate(rho_O_values):
            key = f"rho_T={rho_T}, rho_O={rho_O}"
            heatmap_data[i, j] = results[key]

    # Create the plot
    plt.figure(figsize=(10, 8))
    im = plt.imshow(heatmap_data, cmap='viridis_r')  # viridis_r makes lower values (better) darker

    # Add colorbar
    cbar = plt.colorbar(im)
    #cbar.set_label('L1 Distance (lower is better)')

    # Set labels
    plt.title('POMDP Reconstruction Sensitivity Analysis')
    plt.xlabel('rho_O values')
    plt.ylabel('rho_T values')

    # Set ticks
    plt.xticks(np.arange(len(rho_O_values)), [f"{x:.2f}" for x in rho_O_values])
    plt.yticks(np.arange(len(rho_T_values)), [f"{x:.2f}" for x in rho_T_values])

    # Add text annotations
    for i in range(len(rho_T_values)):
        for j in range(len(rho_O_values)):
            text = plt.text(j, i, f"{heatmap_data[i, j]:.3f}",
                            ha="center", va="center", color="w" if heatmap_data[i, j] > 0.5 else "black")

    plt.tight_layout()
    plt.show()


def visualize_observation_distributions(model, n_states, title_prefix=""):
    """Visualize the observation distributions for each state"""
    plt.figure(figsize=(15, 5))

    for s in range(n_states):
        plt.subplot(1, 3, s + 1)

        # Create a 2D grid of points
        x = np.linspace(0, 1, 100)
        y = np.linspace(0, 1, 100)
        X, Y = np.meshgrid(x, y)

        # Compute the PDF on the grid
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                obs = np.array([X[i, j], Y[i, j]])
                Z[i, j] = model.observation_likelihood(obs, s)

        plt.contourf(X, Y, Z, levels=20, cmap='viridis')
        plt.title(f"{title_prefix} State {s + 1}")
        plt.xlabel("Test Result")
        plt.ylabel("Symptoms")
        plt.colorbar()

    plt.tight_layout()


def run_pomdp_reconstruction():
    """Main experiment function for POMDP reconstruction"""
    # Parameters
    n_states = 3
    n_actions = 2
    seed = 125405  # For reproducibility
    random.seed(seed)
    np.random.seed(seed)

    # Generate training data
    fuzzy_model = build_fuzzy_model()

    observations, actions = _simulate_data(fuzzy_model, 40, 9)
    obs_dim = len(observations[0][0])  # Number of observation dimensions

    #observations = [sublist[:9] for sublist in observations]
    #actions = [sublist[:9] for sublist in actions]

    # Train fuzzy POMDP model
    fuzzy_pomdp = models.trainable.fuzzy_EM.FuzzyPOMDP(
        n_states=n_states,
        n_actions=n_actions,
        obs_dim=obs_dim,
        use_fuzzy=True,  # Set to True if using fuzzy model
        fuzzy_model=fuzzy_model,
        lambda_T=0.5,
        lambda_O=0.5,
        verbose=True,
        obs_var_index=obs_var_index,
        parallel=False,
        hyperparameter_update_method="adaptive",
        epsilon_prior=1e-4
    )

    fuzzy_pomdp.initialize_with_kmeans(observations)

    for a in range(fuzzy_pomdp.n_actions):
        fuzzy_pomdp.transitions[:, a, :] = np.array([[0.5, 0.25, 0.25],
                                                     [0.25, 0.5, 0.25],
                                                     [0.25, 0.25, 0.5]])

    fuzzy_ll = fuzzy_pomdp.fit(
        observations, actions,
        max_iterations=300, tolerance=1e-3
    )

    visualize_observation_distributions_per_state(fuzzy_pomdp, n_states, obs_dim, title_prefix="Fuzzy")
    for s in range(fuzzy_pomdp.n_states):
        for a in range(fuzzy_pomdp.n_actions):
            for s_next in range(fuzzy_pomdp.n_states):
                print(
                    f"Transition probabilities from state {s}, action {a} to state {s_next}: {fuzzy_pomdp.transitions[s, a, s_next]}")
    visualize_covariance_matrices(fuzzy_pomdp, n_states, title_prefix="Fuzzy")
    print(fuzzy_pomdp.transitions)

    return fuzzy_pomdp






def visualize_covariance_matrices(model, n_states, title_prefix=""):
    """
    Visualize the covariance matrices for each state as heatmaps.
    """

    for s in range(n_states):
        plt.figure(figsize=(10, 10))
        sns.heatmap(
            model.obs_covs[s],
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            cbar=True
        )
        plt.title(f"{title_prefix} Covariance Matrix - State {s + 1}")
        plt.xlabel("Observation Dimension")
        plt.ylabel("Observation Dimension")
        plt.tight_layout()
        plt.show()


def visualize_observation_distributions_per_state(model, n_states, obs_dim, title_prefix=""):
    """
    Visualize the observation distributions for each observation dimension across all states.
    """
    # Define the number of rows and columns for subplots
    n_cols = 3  # Number of columns
    n_rows = (obs_dim + n_cols - 1) // n_cols  # Calculate rows dynamically

    plt.figure(figsize=(15, 5 * n_rows))

    for obs_idx in range(obs_dim):
        #check if is not ravu
        if OBS_LIST[obs_idx] != "Ravu":
            plt.subplot(n_rows, n_cols, obs_idx + 1)

            # Create a range of values for the observation dimension
            x = np.linspace(0, 1, 100)

            # Plot the PDF for the observation dimension for each state
            for s in range(n_states):
                y = [norm.pdf(val, loc=model.obs_means[s][obs_idx], scale=np.sqrt(model.obs_covs[s][obs_idx, obs_idx]))
                     for val in x]
                plt.plot(x, y, label=f"State {s}")

            plt.title(f"{title_prefix} Observation {OBS_LIST[obs_idx]}")
            plt.xlabel("Observation Value")
            plt.ylabel("Density")
            plt.legend()

    plt.tight_layout()
    plt.show()


def main():
    """
    Main function to demonstrate POMDP reconstruction using EM algorithm.
    1. Generate data from the original POMDP model
    2. Use EM algorithm to learn/reconstruct the model
    3. Evaluate the reconstruction quality
    """
    best_model = run_pomdp_reconstruction()

    #boostraap_models = run_pomdp_with_bootstrap([], [], n_bootstrap_samples=5, n_states=2, n_actions=2, obs_dim=2)
    return best_model


if __name__ == "__main__":
    learned_pomdp = main()
    #evaluate_fuzzy_reward_prediction(trials=200, horizon=10)
    #sensitivity_results = rho_sensitivity_analysis()
