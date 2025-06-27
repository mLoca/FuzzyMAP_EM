import random
from time import sleep

import numpy as np
import matplotlib.pyplot as plt
import pomdp_py
from scipy.stats import multivariate_normal, norm, entropy

from simpful import FuzzySystem, LinguisticVariable, FuzzyAggregator, GaussianFuzzySet
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from sklearn.metrics import r2_score
from sklearn.mixture import GaussianMixture

from pomdp_EM import PomdpEM

from fuzzy_model import build_fuzzymodel
from continouos_pomdp_example import State, MedicalRewardModel, MedicalTransitionModel, MedAction
from continouos_pomdp_example import plot_observation_distribution, create_continuous_medical_pomdp, STATES, ACTIONS, \
    ContinuousObservationModel

DEFAULT_PARAMS_TRANS = [
    [[0.85, 0.14, 0.01],
     [0.80, 0.15, 0.05]],
    [[0.30, 0.60, 0.10],
     [0.65, 0.35, 0.00]],
    [[0.01, 0.05, 0.94],
     [0.1, 0.65, 0.25]]
]


class FuzzyPOMDP(PomdpEM):
    """
    EM algorithm implementation for POMDPs with:
    - Discrete states
    - Discrete actions
    - Continuous observations (modeled with fuzzy logic)
    """

    def __init__(self, n_states: int, n_actions: int, obs_dim: int, use_fuzzy: bool = False,
                 rho_T: float = 0.05, rho_O: float = 0.05, fuzzy_model=None,
                 verbose: bool = False, fix_transitions=None, fix_observations=None):
        super().__init__(n_states, n_actions, obs_dim, verbose)
        """Initialize the POMDP model with EM capabilities"""

        self.use_fuzzy = use_fuzzy
        self.rho_T = rho_T  # weight parameter for pseudo-counts
        self.rho_O = rho_O  # weight parameter for pseudo-counts in observation model
        self.epsilon_prior = 1e-10
        self.fix_transitions = fix_transitions
        self.fix_observations = fix_observations

        if fix_transitions is not None:
            self.transitions = np.array(fix_transitions, dtype=np.float64)
        # Fuzzy system for observations
        if use_fuzzy:
            if fuzzy_model is None:
                self.fuzzy_model = build_fuzzymodel()
            else:
                self.fuzzy_model = fuzzy_model
        else:
            self.fuzzy_model = None

    def _match_rule_ant(self, rule, action, O_means, state=0):
        """
        Check how much the current observation distribution given a state matches the rule.
        :param rule:
        :return:
        """
        # Extract antecedents from the rule
        antecedents = rule.split("IF")[1].split("THEN")[0].strip().split("AND")
        antecedents = [antecedent.strip() for antecedent in antecedents]

        # Initialize match score
        match_score = 1.0

        # Iterate through each antecedent and compute the match score
        for antecedent in antecedents:
            antecedent = antecedent.replace("(", " ")
            antecedent = antecedent.replace(")", " ")
            variable, term = antecedent.split("IS")
            variable = variable.strip()
            term = term.strip()

            # Get the fuzzy set for the variable and term
            fuzzy_set = self.fuzzy_model.get_fuzzy_set(variable, term)

            # Compute the membership degree of the current observation in the fuzzy set
            if (variable == "test_result"):
                membership_degree = fuzzy_set.get_value(O_means[state][0])
            elif (variable == "symptoms"):
                membership_degree = fuzzy_set.get_value(O_means[state][1])
            else:
                membership_degree = fuzzy_set.get_value(action)

            # Update the match score (you can use different aggregation methods)
            match_score *= membership_degree

        return match_score

    def _match_rule_cons(self, rule, action, O_means, state=0):
        """
        Check how much the consequent of the rule matches the current observation distribution given a state.
        """
        # Extract consequents from the rule
        consequent = rule.split("THEN")[1].strip()
        # Initialize match score
        match_score = 1.0
        # Iterate through each consequent and compute the match score
        if 'next_test' in rule or 'next_symptoms' in rule:
            function_str = consequent.split("IS")[1][1:5]
            term = consequent.split("IS")[0].replace("(", "").strip()
            fun = self.fuzzy_model._outputfunctions[function_str]

            fun = fun.replace('test_result', str(O_means[state][0]))
            fun = fun.replace('symptoms', str(O_means[state][1]))
            fun = fun.replace('action', str(action))
            # Use a safer eval approach
            o_rule_pred = eval(fun)
            return o_rule_pred, term

    def _match_rule_cons_wsim(self, rule, action, obs, state=0):
        """
        Check how much the consequent of the rule matches the current observation distribution given a state.
        """
        # Extract consequents from the rule
        consequent = rule.split("THEN")[1].strip()
        # Initialize match score
        match_score = 1.0
        # Iterate through each consequent and compute the match score
        if 'next_test' in rule or 'next_symptoms' in rule:
            function_str = consequent.split("IS")[1][1:5]
            term = consequent.split("IS")[0].replace("(", "").strip()
            fun = self.fuzzy_model._outputfunctions[function_str]

            fun = fun.replace('test_result', str(obs[0]))
            fun = fun.replace('symptoms', str(obs[1]))
            fun = fun.replace('action', str(action))
            # Use a safer eval approach
            o_rule_pred = eval(fun)
            return o_rule_pred, term

    def _compute_pdf(self, state, obs_pred, obs_term, O_means, O_covs):
        """
        Compute the probability density function for the observation given the state.
        :param state:
        :param obs_pred:
        :return:
        """
        var_o = 0
        mean_o = 0
        if obs_term == "next_test":
            mean_o = O_means[state][0]
            var_o = O_covs[state][0, 0]
        elif obs_term == "next_symptoms":
            mean_o = O_means[state][1]
            var_o = O_covs[state][1, 1]

        # Standard deviation is the square root of variance
        std_o = np.sqrt(var_o)

        # Compute the likelihood of observing 'observed_next_test_value'
        l_part_obs = norm.pdf(obs_pred,
                              loc=mean_o,
                              scale=std_o)
        return l_part_obs

    def _simulate_data_from_current(self, state=0):
        """
        Simulates the observation from the current observation model
        :param state:
        :return:
        """
        # Simulate the observation from the current observation model
        obs = np.random.multivariate_normal(self.obs_means[state], self.obs_covs[state], 50)
        return obs

    def maximization_step(self, observations, actions, gammas, xis):
        """M-step: Update model parameters based on expected sufficient statistics"""
        # Number of sequences
        n_sequences = len(observations)

        # Update initial state probabilities
        self.initial_prob = np.zeros(self.n_states)
        for i in range(n_sequences):
            self.initial_prob += gammas[i][0]
        self.initial_prob /= n_sequences

        # Copy the parameters
        T_k = np.copy(self.transitions)
        O_means_k = np.copy(self.obs_means)
        O_covs_k = np.copy(self.obs_covs)

        if self.fix_transitions is None:
            pseudo_count = np.zeros((self.n_states, self.n_actions, self.n_states))
            # Update transition probabilities
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    # Count transitions for each action
                    numerator = np.zeros(self.n_states)
                    denominator = 0.0

                    for i in range(n_sequences):
                        for t in range(len(actions[i]) - 1):
                            if actions[i][t] == a:
                                # Add counts for state-action transitions
                                for s_prime in range(self.n_states):
                                    numerator[s_prime] += xis[i][t, s, s_prime]
                                denominator += gammas[i][t, s]

                    # PseudoCount
                    if self.use_fuzzy:
                        rules = self.fuzzy_model.get_rules()
                        for s_prime in range(self.n_states):
                            ps_count = 0
                            for r in rules:
                                fr_s = self._match_rule_ant(r, a, O_means_k, s)
                                cons_s, term = self._match_rule_cons(r, a, O_means_k, s)

                                y_cons = np.copy(O_means_k[s])  # Start with mean of premise state
                                if term == "next_test":
                                    y_cons[0] = cons_s
                                else:  # next_symptoms
                                    y_cons[1] = cons_s

                                pdf_s_prime = self.observation_likelihood(y_cons, s_prime)
                                ps_count += fr_s * pdf_s_prime
                            pseudo_count[s, a, s_prime] += ps_count * self.rho_T

                    # Update transition probabilities if we have observations
                    if denominator > 0:
                        numerator += pseudo_count[s, a, :]
                        denominator += np.sum(pseudo_count[s, a, :])
                        self.transitions[s, a, :] = numerator / denominator

        if self.fix_observations is None:
            # Update observation model parameters
            obs_counts = np.zeros(self.n_states)
            new_means = np.zeros((self.n_states, self.obs_dim))
            data_O_sum_gamma_obs_sq = np.zeros((self.n_states, self.obs_dim, self.obs_dim))

            for i in range(n_sequences):
                for t in range(len(observations[i])):
                    obs = observations[i][t]
                    for s in range(self.n_states):
                        obs_counts[s] += gammas[i][t, s]
                        new_means[s] += gammas[i][t, s] * obs
                        data_O_sum_gamma_obs_sq[s] += gammas[i][t, s] * np.outer(obs, obs)

            # Update means and covariances
            ps_count_den = np.zeros(self.n_states)
            ps_count_num_mean = np.zeros((self.n_states, self.obs_dim))
            ps_count_num_com_tmp = np.zeros((self.n_states, self.obs_dim, self.obs_dim))
            if self.use_fuzzy:
                for s in range(self.n_states):
                    rules = self.fuzzy_model.get_rules()
                    for old_s in range(self.n_states):
                        for a in range(self.n_actions):
                            for r in rules:
                                fr_s = self._match_rule_ant(r, a, O_means_k, old_s)
                                prob_new_s = T_k[old_s, a, s]
                                cons_s, term = self._match_rule_cons(r, a, O_means_k, old_s)
                                strength = fr_s * prob_new_s * self.rho_O

                                mean_copy = np.copy(self.obs_means[old_s, :])
                                ps_count_den[s] += strength
                                if term == "next_test":
                                    mean_copy[0] = cons_s
                                    ps_count_num_mean[s, 0] += strength * cons_s
                                else:
                                    mean_copy[1] = cons_s
                                    ps_count_num_mean[s, 1] += strength * cons_s
                                ps_count_num_com_tmp[s, :, :] += strength * np.outer(mean_copy, mean_copy)

            for s in range(self.n_states):
                ps_count_num_cov_sum_matrix_s = ps_count_num_com_tmp[s, :, :]
                if obs_counts[s] > 0:
                    total_weight = obs_counts[s] + ps_count_den[s] + self.epsilon_prior
                    combined_weighted_obs_sum_s = new_means[s] + ps_count_num_mean[s]
                    self.obs_means[s] = combined_weighted_obs_sum_s / total_weight
                    combined_weighted_obs_sq_sum_s = data_O_sum_gamma_obs_sq[s] + ps_count_num_cov_sum_matrix_s
                    E_ooT_s = combined_weighted_obs_sq_sum_s / total_weight
                    mu_s_outer_mu_s = np.outer(self.obs_means[s, :], self.obs_means[s, :])
                    self.obs_covs[s, :, :] = E_ooT_s - mu_s_outer_mu_s
                    self.obs_covs[s, :, :] += 1e-12 * np.eye(self.obs_dim)

        return

    def maximization_step_wsim(self, observations, actions, gammas, xis):
        """M-step: Update model parameters based on expected sufficient statistics"""
        n_sequences = len(observations)
        self.initial_prob = np.zeros(self.n_states)
        for i in range(n_sequences):
            self.initial_prob += gammas[i][0]
        self.initial_prob /= n_sequences

        T_k = np.copy(self.transitions)
        O_means_k = np.copy(self.obs_means)
        O_covs_k = np.copy(self.obs_covs)

        if not self.fix_transitions:
            pseudo_count = np.zeros((self.n_states, self.n_actions, self.n_states))
            for s in range(self.n_states):
                sim_data = self._simulate_data_from_current(s)
                for a in range(self.n_actions):
                    numerator = np.zeros(self.n_states)
                    denominator = 0.0
                    for i in range(n_sequences):
                        for t in range(len(actions[i]) - 1):
                            weight = 1
                            if actions[i][t] == a:
                                for s_prime in range(self.n_states):
                                    numerator[s_prime] += xis[i][t, s, s_prime] * weight
                                denominator += gammas[i][t, s] * weight
                    if self.use_fuzzy:
                        rules = self.fuzzy_model.get_rules()
                        for s_prime in range(self.n_states):
                            ps_count = 0
                            for r in rules:
                                ps_data = np.zeros(len(sim_data))
                                count = 0
                                for d in sim_data:
                                    fr_s = self._match_rule_ant_wsim(r, a, d)
                                    cons_s, term = self._match_rule_cons_wsim(r, a, d)
                                    pdf_s_prime = self._compute_pdf(s_prime, cons_s, term, O_means_k, O_covs_k)
                                    ps_data[count] = fr_s * pdf_s_prime
                                    count += 1
                                ps_count += np.average(ps_data)
                            pseudo_count[s, a, s_prime] += ps_count * self.rho
                    if denominator > 0:
                        numerator += pseudo_count[s, a, :]
                        denominator += np.sum(pseudo_count[s, a, :])
                        self.transitions[s, a, :] = numerator / denominator

        if not self.fix_observations:
            obs_counts = np.zeros(self.n_states)
            new_means = np.zeros((self.n_states, self.obs_dim))
            data_O_sum_gamma_obs_sq = np.zeros((self.n_states, self.obs_dim, self.obs_dim))
            for i in range(n_sequences):
                for t in range(len(observations[i])):
                    obs = observations[i][t]
                    weight = 1
                    for s in range(self.n_states):
                        weighted_gamma = weight * gammas[i][t, s]
                        obs_counts[s] += weighted_gamma
                        new_means[s] += weighted_gamma * obs
                        data_O_sum_gamma_obs_sq[s] += weighted_gamma * np.outer(obs, obs)

            ps_count_den = np.zeros(self.n_states)
            ps_count_num_mean = np.zeros((self.n_states, self.obs_dim))
            ps_count_num_cov = np.zeros((self.n_states, self.obs_dim))
            if self.use_fuzzy:
                for s in range(self.n_states):
                    rules = self.fuzzy_model.get_rules()
                    for old_s in range(self.n_states):
                        sim_data = self._simulate_data_from_current(old_s)
                        for a in range(self.n_actions):
                            strength_data = np.zeros(len(sim_data))
                            cons_data = np.zeros(len(sim_data))
                            for r in rules:
                                index = 0
                                for sim_obs in sim_data:
                                    fr_s = self._match_rule_ant(r, a, O_means_k, old_s)
                                    prob_new_s = T_k[old_s, a, s]
                                    cons_s, term = self._match_rule_cons(r, a, O_means_k, old_s)
                                    strength = fr_s * prob_new_s * self.rho
                                    strength_data[index] = strength
                                    cons_data[index] = cons_s
                                    index += 1
                                strength = np.average(strength_data)
                                cons_s = np.average(cons_data)
                                ps_count_den[s] += strength
                                if term == "next_test":
                                    ps_count_num_mean[s, 0] += strength * cons_s
                                    ps_count_num_cov[s, 0] += strength * (cons_s ** 2)
                                else:
                                    ps_count_num_mean[s, 1] += strength * cons_s
                                    ps_count_num_cov[s, 1] += strength * (cons_s ** 2)

            for s in range(self.n_states):
                ps_count_num_cov_sum_matrix_s = np.diag(ps_count_num_cov[s, :])
                if obs_counts[s] > 0:
                    total_weight = obs_counts[s] + ps_count_den[s] + self.epsilon_prior
                    combined_weighted_obs_sum_s = new_means[s] + ps_count_num_mean[s]
                    self.obs_means[s] = combined_weighted_obs_sum_s / total_weight
                    combined_weighted_obs_sq_sum_s = data_O_sum_gamma_obs_sq[s] + ps_count_num_cov_sum_matrix_s
                    E_ooT_s = combined_weighted_obs_sq_sum_s / total_weight
                    mu_s_outer_mu_s = np.outer(self.obs_means[s, :], self.obs_means[s, :])
                    self.obs_covs[s, :, :] = E_ooT_s - mu_s_outer_mu_s
                    self.obs_covs[s, :, :] += 1e-6 * np.eye(self.obs_dim)

        return


def rho_sensitivity_analysis():
    """Run sensitivity analysis for rho parameter in POMDP reconstruction"""
    # Parameters
    n_trajectories = 10
    trajectory_length = 5
    n_states = 3  # healthy, sick, critical
    n_actions = 2  # wait, treat
    obs_dim = 2  # test_result and symptoms (continuous)
    seed = 12542  # For reproducibility

    # Generate training data
    original_pomdp, observations, actions, true_states, rewards = generate_pomdp_data(
        n_trajectories, trajectory_length, seed=seed
    )

    fuzzy_model = build_fuzzymodel()

    rho_values = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
    results_l1 = {}
    results_kl = {}

    for rho_T in rho_values:
        for rho_O in rho_values:
            print(f"\nRunning POMDP reconstruction with rho_T={rho_T} and rho_O={rho_O}...")
            pomdp_model = FuzzyPOMDP(
                n_states=n_states,
                n_actions=n_actions,
                obs_dim=obs_dim,
                use_fuzzy=True,
                rho_T=rho_T,
                rho_O=rho_O,
                fuzzy_model=fuzzy_model
            )

            log_likelihood = pomdp_model.fit(observations, actions, max_iterations=150, tolerance=1e-5)

            visualize_observation_distributions(pomdp_model, n_states, title_prefix="Fuzzy")
            plt.show()

            default_mapping = {}
            for i, state in enumerate(STATES):
                if state.name == "healthy":
                    default_mapping[state.name] = 2
                elif state.name == "sick":
                    default_mapping[state.name] = 1
                elif state.name == "critical":
                    default_mapping[state.name] = 0

            # Evaluate model distance to the original POMDP
            distances = pomdp_model.compare_state_transitions(
                pomdp_model.transitions,
                original_pomdp.env.transition_model.transitions,
                STATES,
                state_mapping=default_mapping
            )

            str_rho = f"rho_T={rho_T}, rho_O={rho_O}"
            results_l1[str_rho] = distances['l1_distance']

            print(f"Results for rho={str_rho}: {distances}")
            if distances['kl_divergences'] > 6:
                results_kl[str_rho] = 4.0
            else:
                results_kl[str_rho] = distances['kl_divergences']

    plot_sensitivity_results(results_l1)
    plot_sensitivity_results(results_kl)
    return results_l1, results_kl


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


def generate_pomdp_data(n_trajectories, trajectory_length, seed=None):
    """Generate training data from original POMDP model"""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    print("Generating training data...")
    observations = []
    actions = []
    true_states = []
    rewards = []

    for traj in range(n_trajectories):
        # Reset environment for new trajectory
        pomdp = create_continuous_medical_pomdp()
        pomdp.agent.set_belief(pomdp_py.Histogram({
            State("healthy"): 1 / 3, State("sick"): 1 / 3, State("critical"): 1 / 3
        }))

        traj_observations, traj_actions = [], []
        traj_states, traj_rewards = [], []

        for _ in range(trajectory_length):
            # Get current state
            current_state = pomdp.env.state
            traj_states.append(STATES.index(current_state))

            # Select random action (for data collection)
            action = random.choice(ACTIONS)
            action_idx = ACTIONS.index(action)
            traj_actions.append(action_idx)

            # Execute action and get observation
            reward = pomdp.env.state_transition(action, execute=True)
            traj_rewards.append(reward)

            # Get observation with noise
            observation = pomdp.agent.observation_model.sample(pomdp.env.state, action)
            noise = np.random.normal(0, 0.25, size=len(observation))
            noisy_observation = np.clip(observation + noise, 0, 1)
            traj_observations.append(noisy_observation)

        observations.append(np.array(traj_observations))
        actions.append(np.array(traj_actions))
        true_states.append(np.array(traj_states))
        rewards.append(np.array(traj_rewards))

    print(f"Generated {n_trajectories} trajectories of length {trajectory_length}")
    return pomdp, observations, actions, true_states, rewards


def train_pomdp_model(observations, actions, use_fuzzy, n_states, n_actions, obs_dim, max_iterations, tolerance):
    """Train a POMDP model using the provided data"""
    model_type = "Fuzzy" if use_fuzzy else "Standard"
    print(f"Training {model_type} POMDP model...")

    model = FuzzyPOMDP(
        n_states=n_states,
        n_actions=n_actions,
        obs_dim=obs_dim,
        use_fuzzy=use_fuzzy,
        rho_T=0.0,
        rho_O=0.0,
        fix_transitions=DEFAULT_PARAMS_TRANS,  # Set to None to allow learning transitions
    )

    log_likelihood = model.fit(
        observations,
        actions,
        max_iterations=max_iterations,
        tolerance=tolerance
    )

    print(f"{model_type} POMDP training completed. Final log-likelihood: {log_likelihood:.4f}")
    return model, log_likelihood


def visualize_observation_distributions(model, n_states, title_prefix=""):
    """Visualize the observation distributions for each state"""
    plt.figure(figsize=(15, 5))
    state_names = ["Healthy", "Sick", "Critical"]

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
        plt.title(f"{title_prefix} State {state_names[s]}")
        plt.xlabel("Test Result")
        plt.ylabel("Symptoms")
        plt.colorbar()

    plt.tight_layout()


def run_pomdp_reconstruction():
    """Main experiment function for POMDP reconstruction"""
    # Parameters
    n_trajectories = 25
    trajectory_length = 5
    n_states = 3  # healthy, sick, critical
    n_actions = 2  # wait, treat
    obs_dim = 2  # test_result and symptoms (continuous)
    seed = 42  # For reproducibility

    # Generate training data
    original_pomdp, observations, actions, true_states, rewards = generate_pomdp_data(
        n_trajectories, trajectory_length, seed=seed
    )

    # Train fuzzy POMDP model
    fuzzy_pomdp = FuzzyPOMDP(
        n_states=n_states,
        n_actions=n_actions,
        obs_dim=obs_dim,
        use_fuzzy=True,
        fuzzy_model=build_fuzzymodel(),
        rho_T=0.0,
        rho_O=0.1,
        verbose=False,
        fix_transitions=np.array(DEFAULT_PARAMS_TRANS),  # Set to None to allow learning transitions
    )

    fuzzy_ll = fuzzy_pomdp.fit(
        observations, actions,
        max_iterations=200, tolerance=1e-8
    )

    # Visualize fuzzy model results
    visualize_observation_distributions(fuzzy_pomdp, n_states, title_prefix="Fuzzy")
    plt.show()

    # Compare with original model
    fuzzy_pomdp.compare_state_transitions(
        fuzzy_pomdp.transitions,
        original_pomdp.env.transition_model.transitions,
        STATES
    )

    # Train standard POMDP model
    standard_pomdp, standard_ll = train_pomdp_model(
        observations, actions, use_fuzzy=False,
        n_states=n_states, n_actions=n_actions, obs_dim=obs_dim,
        max_iterations=200, tolerance=1e-8
    )

    # Visualize standard model results
    visualize_observation_distributions(standard_pomdp, n_states, title_prefix="Standard")
    plt.show()

    # Compare with original observation distribution
    print("\nComparing with original observation distributions...")
    plot_observation_distribution()

    # Compare standard model with original model
    print("Evaluating standard model against original model...")
    standard_pomdp.compare_state_transitions(
        standard_pomdp.transitions,
        original_pomdp.env.transition_model.transitions,
        STATES
    )

    return fuzzy_pomdp


def main():
    """
    Main function to demonstrate POMDP reconstruction using EM algorithm.
    1. Generate data from the original POMDP model
    2. Use EM algorithm to learn/reconstruct the model
    3. Evaluate the reconstruction quality
    """
    best_model = run_pomdp_reconstruction()
    return best_model


def evaluate_fuzzy_reward_prediction(trials=200, horizon=10):
    """
    Compares the reward predicted by the fuzzy model against the true reward
    and evaluates next-observation prediction accuracy over simulated trajectories.
    """

    np.random.seed(42)
    random.seed(42)

    base_reward_model = MedicalRewardModel()
    fuzzy_model = build_fuzzymodel()
    transition_model = MedicalTransitionModel()
    observation_model = ContinuousObservationModel(STATES, ACTIONS)

    true_next_obs_healthy = []
    true_next_obs_sick = []
    true_next_obs_critical = []

    pred_next_obs_healthy = []
    pred_next_obs_sick = []
    pred_next_obs_critical = []
    all_actions = [MedAction(a) for a in ["wait", "treat"]]
    all_states = [State(s) for s in ["healthy", "sick", "critical"]]

    for _ in range(trials):
        current_state = random.choice(all_states)
        for _ in range(horizon):
            action = random.choice(all_actions)
            # Sample next state and true next observation
            next_state = transition_model.sample(current_state, action)
            actual_obs = observation_model.sample(next_state, action)
            # True reward
            r_true = base_reward_model.sample(current_state, action, next_state)
            # Fuzzy prediction
            action_fm = 0 if action.name == "wait" else 1
            fuzzy_model.set_variable("action_input", action_fm)
            fuzzy_model.set_variable("test_result", actual_obs[0])
            fuzzy_model.set_variable("symptoms", actual_obs[1])
            pred_obs = fuzzy_model.Sugeno_inference(["next_test", "next_symptoms"])
            # Record metrics
            if current_state.name == "healthy":
                true_next_obs_healthy.append(actual_obs)
                pred_next_obs_healthy.append([pred_obs["next_test"], pred_obs["next_symptoms"]])
            elif current_state.name == "sick":
                true_next_obs_sick.append(actual_obs)
                pred_next_obs_sick.append([pred_obs["next_test"], pred_obs["next_symptoms"]])
            elif current_state.name == "critical":
                true_next_obs_critical.append(actual_obs)
                pred_next_obs_critical.append([pred_obs["next_test"], pred_obs["next_symptoms"]])

            current_state = next_state

    # Convert to numpy arrays
    for output in [(true_next_obs_healthy,pred_next_obs_healthy),
                (true_next_obs_sick,pred_next_obs_sick),
                (true_next_obs_critical,pred_next_obs_critical)]:
        true_next_obs = output[0]
        pred_next_obs = output[1]
        test_true = np.array([obs[0] for obs in true_next_obs])
        test_pred = np.array([obs[0] for obs in pred_next_obs])
        symp_true = np.array([obs[1] for obs in true_next_obs])
        symp_pred = np.array([obs[1] for obs in pred_next_obs])

        r2_test = r2_score(test_true, test_pred)
        r2_sym = r2_score(symp_true, symp_pred)

        # Observation prediction accuracy
        #obs_accuracy = np.mean([t == p for t, p in zip(true_next_obs, pred_next_obs)])

        # Print results
        print("\n--- Fuzzy Reward & Observation Prediction Performance ---")
        print(f"Total steps: {trials * horizon}")
        print(f"Test Result R²: {r2_test:.4f}")
        print(f"Symptoms R²: {r2_sym:.4f}")

    return


if __name__ == "__main__":
    learned_pomdp = main()
    #evaluate_fuzzy_reward_prediction(trials=500, horizon=10)
    #sensitivity_results = rho_sensitivity_analysis()
