import random

import numpy as np
from joblib import Parallel, delayed

import models.trainable.fuzzy_EM
from MG.MG_FM import build_fuzzy_model, _simulate_data
from MG.fuzzy_EM import FuzzyPOMDP

obs_var_index = {
    "Teff": 0, "Treg": 1, "B": 2, "GC": 3, "SLPB": 4,
    "LLPC": 5, "IgG": 6, "Complement": 7, "Symptoms": 8, "Inflammation": 9
}
SYMPTOMS_IDX = obs_var_index["Symptoms"]


def run_pomdp_with_bootstrap(n_bootstrap_samples=500, n_states=2, n_actions=2):
    seed = 125405
    random.seed(seed)
    np.random.seed(seed)

    fuzzy_model = build_fuzzy_model()
    observations, actions = _simulate_data(fuzzy_model, 500, 8)
    obs_dim = len(observations[0][0])
    n_sequences = len(observations)

    print(f"Running {n_bootstrap_samples} bootstrap samples in parallel using {12} jobs...")
    trained_models = []
    # Use joblib to run the training in parallel
    trained_models = Parallel(n_jobs=12)(
       delayed(_train_single_bootstrap_model)(
           i, observations, actions, n_sequences, n_states, n_actions, obs_dim, fuzzy_model
       ) for i in range(n_bootstrap_samples))


    #for i in range(n_bootstrap_samples):
    #    model = _train_single_bootstrap_model(
    #        i, observations, actions, n_sequences, n_states, n_actions, obs_dim, fuzzy_model
    #    )
    #   trained_models.append(model)

    print("\nAll bootstrap training finished.")
    _analyze_bootstrap_transitions(trained_models)
    return trained_models


def _train_single_bootstrap_model(sample_index, observations, actions, n_sequences, n_states, n_actions, obs_dim,
                                  fuzzy_model, base_seed=125405):
    """
    Worker function to train a single POMDP model on one bootstrap sample.
    This is designed to be called in parallel.
    """
    print(f"--- Starting Bootstrap Sample {sample_index + 1} ---")
    np.random.seed(base_seed + sample_index)
    # Create a bootstrap sample with replacement
    bootstrap_indices = np.random.choice(n_sequences, size=n_sequences)
    bootstrap_obs = [observations[j] for j in bootstrap_indices]
    bootstrap_actions = [actions[j] for j in bootstrap_indices]

    # Initialize and train a new POMDP model on the bootstrap sample
    bootstrap_pomdp = models.trainable.fuzzy_EM.FuzzyPOMDP(n_states=n_states, n_actions=n_actions, obs_dim=obs_dim,
                                                           use_fuzzy=True, fuzzy_model=fuzzy_model, lambda_T=0.50,
                                                           lambda_O=0.50, hyperparameter_update_method="static",
                                                           parallel=False, obs_var_index=obs_var_index,
                                                           epsilon_prior=1e-3, ensure_psd=True)

    bootstrap_pomdp.initialize_with_kmeans(bootstrap_obs)

    bootstrap_pomdp.fit(
        bootstrap_obs,
        bootstrap_actions,
        max_iterations=100,
        tolerance=1e-2
    )

    print(f"--- Finished Bootstrap Sample {sample_index + 1} ---")
    return bootstrap_pomdp


def _analyze_bootstrap_transitions(trained_models):
    """
    Analyzes specific transitions from State 2  to State 1  and State 0 .

    Args:
        trained_models (list): A list of trained FuzzyPOMDP models from the bootstrap.
    """
    # Store the probabilities for staying in state 0 for each action
    trans_2_to_1_a0 = []  # Action 0 (Wait)
    trans_2_to_1_a1 = []  # Action 1 (Ravu)
    trans_2_to_0_a0 = []  # Action 0 (Wait)
    trans_2_to_0_a1 = []  # Action 1 (Ravu)

    for model in trained_models:
        symptoms_means = model.obs_means[:, SYMPTOMS_IDX]
        sorted_indices = np.argsort(symptoms_means)

        idx_healthy = sorted_indices[0]
        idx_sick = sorted_indices[1]
        idx_critical = sorted_indices[2]

        # Critical to Sick
        p_21_a0 = model.transitions[idx_critical, 0, idx_sick]
        trans_2_to_1_a0.append(p_21_a0)
        p_21_a1 = model.transitions[idx_critical, 1, idx_sick]
        trans_2_to_1_a1.append(p_21_a1)

        # Critical to Healthy
        p_20_a0 = model.transitions[idx_critical, 0, idx_healthy]
        trans_2_to_0_a0.append(p_20_a0)
        p_20_a1 = model.transitions[idx_critical, 1, idx_healthy]
        trans_2_to_0_a1.append(p_20_a1)

    _print_stats("Critical -> Sick", trans_2_to_1_a0, trans_2_to_1_a1)

    # Analyze 2 -> 0
    _print_stats("Critical -> Healthy", trans_2_to_0_a0, trans_2_to_0_a1)


def _print_stats(name, data_a0, data_a1):
    mean_a0 = np.mean(data_a0)
    ci_a0 = np.percentile(data_a0, [2.5, 97.5])

    mean_a1 = np.mean(data_a1)
    ci_a1 = np.percentile(data_a1, [2.5, 97.5])

    diffs = np.array(data_a1) - np.array(data_a0)
    mean_diff = np.mean(diffs)
    ci_diff = np.percentile(diffs, [2.5, 97.5])

    print(f"\n>>> Transition: {name}")
    print(f"  Action 0 (Wait):  Mean={mean_a0:.4f}, 95% CI=[{ci_a0[0]:.4f}, {ci_a0[1]:.4f}]")
    print(f"  Action 1 (Treat): Mean={mean_a1:.4f}, 95% CI=[{ci_a1[0]:.4f}, {ci_a1[1]:.4f}]")
    print(f"  Difference (A1-A0): Mean={mean_diff:.4f}, 95% CI=[{ci_diff[0]:.4f}, {ci_diff[1]:.4f}]")


def main():
    """
    Main function to demonstrate POMDP reconstruction using EM algorithm.
    1. Generate data from the original POMDP model
    2. Use EM algorithm to learn/reconstruct the model
    3. Evaluate the reconstruction quality
    """
    # best_model = run_pomdp_reconstruction()

    boostraap_models = run_pomdp_with_bootstrap(n_bootstrap_samples=40, n_states=3, n_actions=2)
    return boostraap_models


if __name__ == "__main__":
    learned_pomdp = main()
    # evaluate_fuzzy_reward_prediction(trials=200, horizon=10)
    # sensitivity_results = rho_sensitivity_analysis()
