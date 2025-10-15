import random

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import norm, multivariate_normal
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, root_mean_squared_error

import utils
from continouos_pomdp_example import State, MedicalRewardModel, MedicalTransitionModel, MedAction
from continouos_pomdp_example import (generate_pomdp_data,
                                      STATES, ACTIONS, ContinuousObservationModel)
from fuzzy_model import build_fuzzymodel
from pomdp_EM import PomdpEM

from MG.MG_FM import _simulate_data, build_fuzzy_model


OBS_LIST = [
    "Teff", "Treg", "B", "GC", "SLPB", "LLPC", "IgG", "Complement","Symptoms", "Inflammation",
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
                 verbose: bool = False, fix_transitions=None, fix_observations=None,
                 fixed_observation_states=None):
        super().__init__(n_states, n_actions, obs_dim, verbose)
        """Initialize the POMDP model with EM capabilities"""

        self.use_fuzzy = use_fuzzy
        self.rho_T = rho_T  # weight parameter for pseudo-counts
        self.rho_O = rho_O  # weight parameter for pseudo-counts in observation model
        self.epsilon_prior = 1e-10
        self.fix_transitions = fix_transitions
        self.fix_observations = fix_observations
        self.fixed_observation_states = fixed_observation_states if fixed_observation_states is not None else []

        if fix_transitions is not None:
            self.transitions = np.array(fix_transitions, dtype=np.float64)

        if fix_observations is not None:
            # Convert observation parameters to means and covariances
            tmp_observation = utils.from_beta_to_multivariate_normal(fix_observations)
            if len(self.fixed_observation_states) > 2:
                self.obs_means = np.array(tmp_observation['means'], dtype=np.float64)
                self.obs_covs = np.array(tmp_observation['covariances'], dtype=np.float64)
            elif len(self.fixed_observation_states) > 0:
                for s in range(self.n_states):
                    if s in self.fixed_observation_states:
                        self.obs_means[s] = np.array(tmp_observation['means'][s], dtype=np.float64)
                        self.obs_covs[s] = np.array(tmp_observation['covariances'][s], dtype=np.float64)


        # Fuzzy system for observations
        if use_fuzzy:
            if fuzzy_model is None:
                self.fuzzy_model = build_fuzzymodel()
            else:
                self.fuzzy_model = fuzzy_model
        else:
            self.fuzzy_model = None

    def _get_variable_mapping(self, variable_name,action):

        variable_mapping = {
            "Teff": 0,
            "Treg": 1,
            "B": 2,
            "GC": 3,
            "SLPB": 4,
            "LLPC": 5,
            "IgG": 6,
            "Complement": 7,
            "Symptoms": 8,
            "Inflammation": 9,
        }

        return_value = variable_mapping.get(variable_name, None)
        if action is not None and return_value is None:
            return 10

        return return_value

    def initialize_with_kmeans(self, observations):
        """
        Initializes observation model means using K-Means clustering.
        This provides a better starting point for the EM algorithm.
        """
        # Flatten the list of observation sequences into a single array
        all_obs = np.vstack([obs_seq for obs_seq in observations])

        # Use K-Means to find cluster centers
        kmeans = KMeans(n_clusters=self.n_states, random_state=0, n_init=10)
        kmeans.fit(all_obs)

        # Assign the cluster centers as the initial means for the observation model
        self.obs_means = kmeans.cluster_centers_
        print("Initialized observation means with K-Means.")

    def initialize_transitions_with_fuzzy_rules(self):
        """
        Initializes the transition matrix using the fuzzy rules.
        This should be called after the observation model is initialized.
        """
        if not self.use_fuzzy or self.fuzzy_model is None:
            print("Fuzzy model not in use. Skipping fuzzy transition initialization.")
            return

        print("Initializing transition matrix with fuzzy rules...")
        initial_transitions = np.zeros((self.n_states, self.n_actions, self.n_states))
        rules = self.fuzzy_model.get_rules()

        for s in range(self.n_states):
            for a in range(self.n_actions):
                for s_prime in range(self.n_states):
                    transition_strength = 0.0
                    for r in rules:
                        # Match antecedent with the current state's observation mean
                        fr_s = self._match_rule_ant(r, a, self.obs_means, s)

                        # Get the consequent's predicted value
                        cons_s, term = self._match_rule_cons(r, a, self.obs_means, s)

                        # Create a hypothetical next observation based on the rule
                        y_cons = np.copy(self.obs_means[s])
                        mapping_value = self._get_variable_mapping(term, None)
                        y_cons[mapping_value] = cons_s

                        # Calculate likelihood of this observation under the next state's model
                        pdf_s_prime = self.observation_likelihood(y_cons, s_prime)
                        transition_strength += fr_s * pdf_s_prime

                    initial_transitions[s, a, s_prime] = transition_strength

                # Normalize the probabilities for the (s, a) pair
                total_strength = np.sum(initial_transitions[s, a, :])
                if total_strength > 1e-9:
                    initial_transitions[s, a, :] /= total_strength
                else:
                    # Fallback to uniform if no rule applies
                    initial_transitions[s, a, :] = 1.0 / self.n_states

        self.transitions = initial_transitions
        print(self.transitions)
        print("Transition matrix initialized.")
    def _match_rule_ant(self, rule, action, O_means, state=0):
        """
        Check how much the current observation distribution given a state matches the rule.
        :param rule:
        :return:
        """
        isAnd = True
        # Extract antecedents from the rule
        if "AND" not in rule:
            antecedents = rule.split("IF")[1].split("THEN")[0].strip().split("OR")
            isAnd = False
        else:
            antecedents = rule.split("IF")[1].split("THEN")[0].strip().split("AND")
        antecedents = [antecedent.strip() for antecedent in antecedents]

        # Initialize match score
        match_score = 1.0
        if not isAnd:
            match_score = 0.0

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
            mapping_value = self._get_variable_mapping(variable, action)

            if mapping_value == 10:
                membership_degree = fuzzy_set.get_value(action)
            else:
                membership_degree = fuzzy_set.get_value(O_means[state][mapping_value])

            # Update the match score (you can use different aggregation methods)
            if isAnd:
                match_score *= membership_degree
            else:
                match_score = max(match_score, membership_degree)

        return match_score

    def _match_rule_ant_wsim(self, rule, action, obs):
        """
        Check how much the current observation distribution given a state matches the rule.
        :param rule:
        :return:
        """
        isAnd = True
        # Extract antecedents from the rule
        #Check if AND is present in the rule
        if rule.contains("OR"):
            antecedents = rule.split("IF")[1].split("THEN")[0].strip().split("OR")
            isAnd = False
        else:
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

            mapping_value = self._get_variable_mapping(variable, action)

            membership_degree = fuzzy_set.get_value(obs[mapping_value])


            # Update the match score (you can use different aggregation methods)
            if isAnd:
                match_score *= membership_degree
            else:
                match_score = max(match_score, membership_degree)

        return match_score

    def _fun_replace(self, fun, O_means, action, state):
        for o in OBS_LIST:
            mapping_value = self._get_variable_mapping(o, action)
            fun = fun.replace(o, str(O_means[state][mapping_value]))

    def _match_rule_cons(self, rule, action, O_means, state=0):
        """
        Check how much the consequent of the rule matches the current observation distribution given a state.
        NB: in this case is a mandani model
        """
        # Extract consequents from the rule
        consequent = rule.split("THEN")[1].strip()
        consequent = consequent.replace("(", " ")
        consequent = consequent.replace(")", " ")

        # Iterate through each consequent and compute the match score
        variable, term = consequent.split("IS")
        variable = variable.strip()
        term = term.strip()

        # Get the fuzzy set for the variable and term
        fuzzy_set = self.fuzzy_model.get_fuzzy_set(variable, term)

        mapping_value = self._get_variable_mapping(variable, action)

        membership_degree = fuzzy_set.get_value(O_means[state][mapping_value])

        # Use a safer eval approach
        o_rule_pred = membership_degree
        return o_rule_pred, term

    def _compute_pdf(self, state, obs_pred, obs_term, O_means, O_covs):
        """
        Compute the probability density function for the observation given the state.
        :param state:
        :param obs_pred:
        :return:
        """

        mapping_value = self._get_variable_mapping(obs_term, None)

        mean_o = O_means[state][mapping_value]
        var_o = O_covs[state][mapping_value, mapping_value]

        # Standard deviation is the square root of variance
        std_o = np.sqrt(var_o)

        # Compute the likelihood of observing 'observed_next_test_value'
        l_part_obs = norm.pdf(obs_pred,
                              loc=mean_o,
                              scale=std_o)
        return l_part_obs

    def observation_likelihood(self, obs: np.ndarray, state: int) -> float:
        mu = self.obs_means[state]
        Sigma = self.obs_covs[state]
        # Symmetrize and add tiny jitter for numerical stability
        Sigma = 0.5 * (Sigma + Sigma.T)
        Sigma = Sigma + 1e-8 * np.eye(self.obs_dim, dtype=Sigma.dtype)
        return float(multivariate_normal(mean=mu, cov=Sigma, allow_singular=True).pdf(obs))

    def _simulate_data_from_current(self, state=0, size=100):
        """
        Simulates the observation from the current observation model
        :param state:
        :return:
        """
        # Simulate the observation from the current observation model
        obs = np.random.multivariate_normal(self.obs_means[state], self.obs_covs[state], size)
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
                        try:
                            rules = self.fuzzy_model.get_rules()
                            for s_prime in range(self.n_states):
                                ps_count = 0
                                for r in rules:
                                    fr_s = self._match_rule_ant(r, a, O_means_k, s)
                                    cons_s, term = self._match_rule_cons(r, a, O_means_k, s)

                                    y_cons = np.copy(O_means_k[s])# Start with mean of premise state
                                    mapping_value = self._get_variable_mapping(term, None)

                                    y_cons[mapping_value] = cons_s

                                    pdf_s_prime = self.observation_likelihood(y_cons, s_prime)
                                    ps_count += fr_s * pdf_s_prime
                                pseudo_count[s, a, s_prime] += ps_count * self.rho_T
                        except Exception as e:
                            print(f"Error matching fuzzy rules: {e}")
                            break
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

                                mapping_value = self._get_variable_mapping(term, None)

                                mean_copy[mapping_value] = cons_s
                                ps_count_num_mean[s, mapping_value] += strength * cons_s

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

        print("M-step completed. Updated parameters.")
        return

    def maximization_step_AA(self, observations, actions, gammas, xis):
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

        if (self.fix_observations is None or
            (2 > len(self.fixed_observation_states) > 0)):

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
                if s not in self.fixed_observation_states:
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
                            if actions[i][t] == a:
                                for s_prime in range(self.n_states):
                                    numerator[s_prime] += xis[i][t, s, s_prime]
                                denominator += gammas[i][t, s]

                    if self.use_fuzzy:
                        rules = self.fuzzy_model.get_rules()
                        for s_prime in range(self.n_states):
                            ps_count = 0
                            for r in rules:
                                ps_data = np.zeros(len(sim_data))
                                count = 0
                                cons_s, term = self._match_rule_cons(r, a, O_means_k, s)
                                for d in sim_data:
                                    fr_s = self._match_rule_ant_wsim(r, a, d)
                                    #cons_s, term = self._match_rule_cons(r, a, s)
                                    y_cons = np.copy(O_means_k[s])  # Start with mean of premise state
                                    if term == "next_test":
                                        y_cons[0] = cons_s
                                    else:  # next_symptoms
                                        y_cons[1] = cons_s

                                    pdf_s_prime = self.observation_likelihood(y_cons, s_prime)
                                    ps_data[count] = fr_s * pdf_s_prime
                                    count += 1
                                ps_count += np.average(ps_data)
                            pseudo_count[s, a, s_prime] += ps_count * self.rho_T
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
                                cons_s, term = self._match_rule_cons(r, a, O_means_k, old_s)
                                for sim_obs in sim_data:
                                    fr_s = self._match_rule_ant_wsim(r, a, sim_obs)
                                    prob_new_s = T_k[old_s, a, s]
                                    #cons_s, term = self._match_rule_cons(r, a, O_means_k, old_s)
                                    strength = fr_s * prob_new_s * self.rho_O
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


def train_pomdp_model(observations, actions, use_fuzzy, n_states, n_actions, obs_dim, max_iterations, tolerance):
    """Train a POMDP model using the provided data"""
    print(f"Training standard POMDP model...")

    model = FuzzyPOMDP(
        n_states=n_states,
        n_actions=n_actions,
        obs_dim=obs_dim,
        use_fuzzy=use_fuzzy,
        rho_T=0.0,
        rho_O=0.0,
        fix_transitions=None,  # Set to None to allow learning transitions
    )

    log_likelihood = model.fit(
        observations,
        actions,
        max_iterations=max_iterations,
        tolerance=tolerance
    )

    print(f"Standard POMDP training completed. Final log-likelihood: {log_likelihood:.4f}")
    return model, log_likelihood


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
    n_states = 2
    n_actions = 2
    seed = 125405  # For reproducibility
    random.seed(seed)
    np.random.seed(seed)

    # Generate training data
    fuzzy_model = build_fuzzy_model()

    observations, actions = _simulate_data(fuzzy_model, 40, 9)
    obs_dim = len(observations[0][0])  # Number of observation dimensions

    # Train fuzzy POMDP model
    fuzzy_pomdp = FuzzyPOMDP(
        n_states=n_states,
        n_actions=n_actions,
        obs_dim=obs_dim,
        use_fuzzy=True,  # Set to True if using fuzzy model
        fuzzy_model=fuzzy_model,
        rho_T=0.05,
        rho_O=0.05,
        verbose=True,
        # fix_transitions=np.array(DEFAULT_PARAMS_TRANS),  # Set to None to allow learning transitions
    )

    fuzzy_pomdp.initialize_with_kmeans(observations)
    #fuzzy_pomdp.initialize_transitions_with_fuzzy_rules()

    #visualize_observation_distributions_per_state(fuzzy_pomdp, n_states, obs_dim, title_prefix="Fuzzy")
    #print(f"Fuzzy POMDP transitions:\n{fuzzy_pomdp.transitions}")
    fuzzy_ll = fuzzy_pomdp.fit(
        observations, actions,
        max_iterations=9, tolerance=1e-3
    )

    fuzzy_pomdp.rho_O = 0.000
    fuzzy_pomdp.rho_T = 0.000

    fuzzy_ll = fuzzy_pomdp.fit(
        observations, actions,
        max_iterations=1, tolerance=1e-3
    )




    for s in range(fuzzy_pomdp.n_states):
        for a in range(fuzzy_pomdp.n_actions):
            print(f"Transition probabilities from state {s}, action {a}: {fuzzy_pomdp.transitions[s, a, :]}")
    # Visualize fuzzy model results
    #visualize_observation_distributions(fuzzy_pomdp, n_states, title_prefix="Fuzzy")
    #plt.show()

    # Compare with original model
    #fuzzy_pomdp.compare_state_transitions(
    #    fuzzy_pomdp.transitions,
    #    original_pomdp.env.transition_model.transitions,
    #    original_pomdp.agent.observation_model,
    #    STATES
    #)

    visualize_observation_distributions_per_state(fuzzy_pomdp, n_states, obs_dim, title_prefix="Fuzzy")

    visualize_covariance_matrices(fuzzy_pomdp, n_states, title_prefix="Fuzzy")

    return fuzzy_pomdp

def run_pomdp_with_bootstrap(observations, actions, n_bootstrap_samples=10, n_states=2, n_actions=2, obs_dim=10):
    seed = 125405
    random.seed(seed)
    np.random.seed(seed)

    fuzzy_model = build_fuzzy_model()
    observations, actions = _simulate_data(fuzzy_model, 40, 9)
    obs_dim = len(observations[0][0])
    n_sequences = len(observations)

    print(f"Running {n_bootstrap_samples} bootstrap samples in parallel using {5} jobs...")

    # Use joblib to run the training in parallel
    trained_models = Parallel(n_jobs=5)(
        delayed(_train_single_bootstrap_model)(
            i, observations, actions, n_sequences, n_states, n_actions, obs_dim, fuzzy_model
        ) for i in range(n_bootstrap_samples)
    )

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
        rho_T=0.05,
        rho_O=0.05,
        verbose=False
    )

    bootstrap_pomdp.initialize_with_kmeans(bootstrap_obs)
    bootstrap_pomdp.fit(
        bootstrap_obs,
        bootstrap_actions,
        max_iterations=10,
        tolerance=1e-3
    )

    # Second fitting stage as in the original code
    bootstrap_pomdp.rho_O = 0.000
    bootstrap_pomdp.rho_T = 0.000
    bootstrap_pomdp.fit(
        observations, actions,
        max_iterations=1, tolerance=1e-3
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
                plt.plot(x, y, label=f"State {s + 1}")

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
    #best_model = run_pomdp_reconstruction()

    boostraap_models = run_pomdp_with_bootstrap([], [], n_bootstrap_samples=5, n_states=2, n_actions=2, obs_dim=2)
    return boostraap_models


if __name__ == "__main__":
    learned_pomdp = main()
    #evaluate_fuzzy_reward_prediction(trials=200, horizon=10)
    #sensitivity_results = rho_sensitivity_analysis()
