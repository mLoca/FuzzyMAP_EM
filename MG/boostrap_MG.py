import random

import numpy as np
from joblib import Parallel, delayed

from MG.MG_FM import build_fuzzy_model, _simulate_data
from MG.fuzzy_EM import FuzzyPOMDP


def run_pomdp_with_bootstrap(observations, actions, n_bootstrap_samples=500, n_states=2, n_actions=2, obs_dim=10):
    seed = 125405
    random.seed(seed)
    np.random.seed(seed)

    fuzzy_model = build_fuzzy_model()
    observations, actions = _simulate_data(fuzzy_model, 90, 5)
    obs_dim = len(observations[0][0])
    n_sequences = len(observations)


    print(f"Running {n_bootstrap_samples} bootstrap samples in parallel using {12} jobs...")
    trained_models = []
    # Use joblib to run the training in parallel
    #trained_models = Parallel(n_jobs=12)(
    #    delayed(_train_single_bootstrap_model)(
    #        i, observations, actions, n_sequences, n_states, n_actions, obs_dim, fuzzy_model
    #    ) for i in range(n_bootstrap_samples)
    #)

    for i in range(n_bootstrap_samples):
        model = _train_single_bootstrap_model(
            i, observations, actions, n_sequences, n_states, n_actions, obs_dim, fuzzy_model
        )
        trained_models.append(model)

    print("\nAll bootstrap training finished.")
    analyze_bootstrap_transitions(trained_models)
    return trained_models

def _train_single_bootstrap_model(sample_index, observations, actions, n_sequences, n_states, n_actions, obs_dim, fuzzy_model):
    """
    Worker function to train a single POMDP model on one bootstrap sample.
    This is designed to be called in parallel.
    """
    print(f"--- Starting Bootstrap Sample {sample_index + 1} ---")

    # Create a bootstrap sample with replacement
    bootstrap_indices = np.random.choice(n_sequences, size=n_sequences, replace=True)
    bootstrap_obs = [observations[j] for j in bootstrap_indices]
    bootstrap_actions = [actions[j] for j in bootstrap_indices]

    # Initialize and train a new POMDP model on the bootstrap sample
    bootstrap_pomdp = FuzzyPOMDP(
        n_states=n_states,
        n_actions=n_actions,
        obs_dim=obs_dim,
        use_fuzzy=True,
        fuzzy_model=fuzzy_model,
        rho_T=0.10,
        rho_O=0.05,
        verbose=False,
        parallel=False
    )

    bootstrap_pomdp.initialize_with_kmeans(bootstrap_obs)
    bootstrap_pomdp.fit(
        bootstrap_obs,
        bootstrap_actions,
        max_iterations=100,
        tolerance=1e-3
    )

    print(f"--- Finished Bootstrap Sample {sample_index + 1} ---")
    return bootstrap_pomdp


def analyze_bootstrap_transitions(trained_models):
    """
    Analyzes the transition probabilities of staying in state 0 from a list of bootstrapped models.

    Args:
        trained_models (list): A list of trained FuzzyPOMDP models from the bootstrap.
    """
    # Store the probabilities for staying in state 0 for each action
    stay_in_s0_a0 = []
    stay_in_s0_a1 = []
    diff = []

    for model in trained_models:
        # P(s'=0 | s=0, a=0)
        prob_a0 = model.transitions[0, 0, 0]
        stay_in_s0_a0.append(prob_a0)

        # P(s'=0 | s=0, a=1)
        prob_a1 = model.transitions[0, 1, 0]
        stay_in_s0_a1.append(prob_a1)

        diff_model = prob_a1 - prob_a0
        diff.append(diff_model)

    # Calculate statistics
    mean_a0 = np.mean(stay_in_s0_a0)
    std_a0 = np.std(stay_in_s0_a0)
    mean_a1 = np.mean(stay_in_s0_a1)
    std_a1 = np.std(stay_in_s0_a1)
    diff_mean = np.mean(diff)
    diff_std = np.std(diff)

    ci_a0_lower = np.percentile(stay_in_s0_a0, 2.5)
    ci_a0_upper = np.percentile(stay_in_s0_a0, 97.5)
    ci_a1_lower = np.percentile(stay_in_s0_a1, 2.5)
    ci_a1_upper = np.percentile(stay_in_s0_a1, 97.5)
    diff_ci_lower = np.percentile(diff, 2.5)
    diff_ci_upper = np.percentile(diff, 97.5)

    print("\n--- Bootstrap Analysis for Staying in State 0 ---")
    print(f"Action 0 -> P(s'=0 | s=0, a=0):")
    print(f"  Mean={mean_a0:.4f}, Std Dev={std_a0:.4f}")
    print(f"  95% CI: [{ci_a0_lower:.4f}, {ci_a0_upper:.4f}]")

    print(f"\nAction 1 -> P(s'=0 | s=0, a=1):")
    print(f"  Mean={mean_a1:.4f}, Std Dev={std_a1:.4f}")
    print(f"  95% CI: [{ci_a1_lower:.4f}, {ci_a1_upper:.4f}]")

    print(f"\nDifference (Action 0 - Action 1):")
    print(f"  Mean Difference={diff_mean:.4f}, Std Dev={diff_std:.4f}")
    print(f"  95% CI: [{diff_ci_lower:.4f}, {diff_ci_upper:.4f}]")

def main():
    """
    Main function to demonstrate POMDP reconstruction using EM algorithm.
    1. Generate data from the original POMDP model
    2. Use EM algorithm to learn/reconstruct the model
    3. Evaluate the reconstruction quality
    """
    # best_model = run_pomdp_reconstruction()

    boostraap_models = run_pomdp_with_bootstrap([], [],
                                                n_bootstrap_samples=150, n_states=2, n_actions=2, obs_dim=2)
    return boostraap_models

if __name__ == "__main__":
    learned_pomdp = main()
    # evaluate_fuzzy_reward_prediction(trials=200, horizon=10)
    # sensitivity_results = rho_sensitivity_analysis()