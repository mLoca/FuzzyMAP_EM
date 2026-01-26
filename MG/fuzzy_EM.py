import random

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from matplotlib.path import Path
import seaborn as sns
import numpy as np
from scipy.stats import norm

import models.trainable.fuzzy_EM

from MG.MG_FM import _simulate_data, build_fuzzy_model
import plotly.graph_objects as go
from plotly.subplots import make_subplots

OBS_LIST = [
    "Teff", "Treg", "B", "GC", "SLPB", "LLPC", "IgG", "Complement", "Symptoms", "Inflammation",
]
obs_var_index = {
    "Teff": 0, "Treg": 1, "B": 2, "GC": 3, "SLPB": 4,
    "LLPC": 5, "IgG": 6, "Complement": 7, "Symptoms": 8, "Inflammation": 9
}

def run_pomdp_reconstruction(save_data = True, save_probabilities = True):
    """Main experiment function for POMDP reconstruction"""
    # Parameters
    n_states = 3
    n_actions = 2
    seed = 125405
    random.seed(seed)
    np.random.seed(seed)

    # Generate training data
    fuzzy_model = build_fuzzy_model()
    observations, actions = _simulate_data(fuzzy_model, 200, 8)
    obs_dim = len(observations[0][0])

    # Train fuzzy POMDP model
    fuzzy_pomdp = models.trainable.fuzzy_EM.FuzzyPOMDP(n_states=n_states, n_actions=n_actions, obs_dim=obs_dim,
                                                       use_fuzzy=True, fuzzy_model=fuzzy_model, lambda_T=2,
                                                       lambda_O=2, verbose=True, obs_var_index=obs_var_index,
                                                       parallel=False, hyperparameter_update_method="adaptive",
                                                       ensure_psd=True, fix_observations=None)

    fuzzy_pomdp.initialize_with_kmeans(observations)

    for a in range(fuzzy_pomdp.n_actions):
        fuzzy_pomdp.transitions[:, a, :] = np.array([[0.55, 0.25, 0.2],
                                                     [0.2, 0.6, 0.2],
                                                     [0.2, 0.25, 0.55]])

    fuzzy_pomdp.obs_covs = fuzzy_pomdp.obs_covs * 0.03
    fuzzy_pomdp.transition_inertia = 40
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
    visualize_observation_distributions_per_state(fuzzy_pomdp, n_states, obs_dim, title_prefix="Fuzzy",
                                                  index_to_exclude=[0, 1, 2, 3, 4, 5, 9])

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


def visualize_observation_distributions_per_state(model, n_states, obs_dim, title_prefix="", index_to_exclude=None):
    """
    Visualize the observation distributions for each observation dimension across all states.
    """
    # Define the number of rows and columns for subplots
    # Order the state by the means of the "Symptoms" observation
    if index_to_exclude is None:
        index_to_exclude = []


    n_cols = 3  # Number of columns
    n_rows = (obs_dim - len(index_to_exclude) + n_cols - 1) // n_cols  # Calculate rows dynamically

    plt.figure(figsize=(15, 5 * n_rows))

    for obs_idx in range(obs_dim):
        #check if is not ravu
        if OBS_LIST[obs_idx] != "Ravu" and obs_idx not in index_to_exclude:
            obs_plt_idx = obs_idx - sum(1 for i in index_to_exclude if i < obs_idx)
            plt.subplot(n_rows, n_cols, obs_plt_idx + 1)

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







if __name__ == "__main__":
    run_pomdp_reconstruction()
