import random
from time import sleep

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from scipy.special import digamma
from scipy.stats import norm
from sklearn.metrics import r2_score, root_mean_squared_error

from utils import utils
from continouos_pomdp_example import State, MedicalRewardModel, MedicalTransitionModel, MedAction
from continouos_pomdp_example import (generate_pomdp_data,
                                      STATES, ACTIONS, ContinuousObservationModel)
from fuzzy.fuzzy_model import build_fuzzymodel
from utils.utils import _prob_sum, multidigamma
from .pomdp_EM import PomdpEM

DEFAULT_PARAMS_TRANS = [
    [[0.85, 0.14, 0.01],
     [0.80, 0.15, 0.05]],
    [[0.30, 0.60, 0.10],
     [0.65, 0.35, 0.00]],
    [[0.01, 0.05, 0.94],
     [0.1, 0.65, 0.25]]
]

DEFAULT_OBS_PARAMS = {
    #beta and alpha parameters for test and symptoms observations
    "healthy": (2, 6, 2, 6),
    "sick": (2.8, 4, 7, 3),
    "critical": (6, 2, 6, 2)
}


#TODO: refactor to generalize the wsim functions
class FuzzyPOMDP(PomdpEM):
    """
    EM algorithm implementation for POMDPs with:
    - Discrete states
    - Discrete actions
    - Continuous observations (modeled with fuzzy logic)
    """

    def __init__(self, n_states: int, n_actions: int, obs_dim: int, seed=42, use_fuzzy: bool = False,
                 lambda_T=0.05, lambda_O=0.05, fuzzy_model=None,
                 verbose=False, fix_transitions=None, fix_observations=None,
                 fixed_observation_states=None, obs_var_index=None, parallel=True,
                 use_adaptive_lambda=False,
                 integration_method="mean",  # 'mean' or 'montecarlo'
                 mc_samples=100,
                 hyperparameter_update_method="static",  # 'static', 'adaptive', 'empirical_bayes'
                 eb_learning_rate=0.00001,
                 alpha_ah=0.1,
                 lambda_min=0.0,
                 epsilon_prior=1e-4
                 ):
        super().__init__(n_states, n_actions, obs_dim, verbose, parallel=parallel, seed=seed,
                         epsilon_prior=epsilon_prior)

        """Initialize the POMDP model with EM capabilities"""

        self.use_fuzzy = use_fuzzy
        self.lambda_T = np.ones(self.n_states) * lambda_T  # weight parameter for pseudo-counts
        self.lambda_O = np.ones(self.n_states) * lambda_O  # weight parameter for pseudo-counts in observation model
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

        # Integration Strategy
        self.integration_method = integration_method  # 'mean' or 'monte_carlo'
        self.mc_samples = mc_samples

        # Hyperparameters for Empirical Bayes
        self.hp_method = hyperparameter_update_method
        self.eb_lr = eb_learning_rate

        #Adaptive Learning
        self.alpha_ah = alpha_ah
        self.prev_fuzzy_counts_T = None
        self.prev_fuzzy_counts_O = None
        self.improvement = None

        # Fuzzy system for observations
        if use_fuzzy:
            self.use_adaptive_lambda = use_adaptive_lambda
            self.lambda_min = lambda_min
            if fuzzy_model is None:
                self.fuzzy_model = build_fuzzymodel(seed=seed)
            else:
                self.fuzzy_model = fuzzy_model

        self.obs_var_index = obs_var_index if obs_var_index is not None else {"test_result": 0, "symptoms": 1}

    def _match_rule_ant(self, rule, action, O_means, state=0):
        """
        Check how much the current observation distribution given a state matches the rule.
        :param rule:
        :return:
        """
        if self.integration_method == "montecarlo":
            points = np.random.multivariate_normal(
                mean=self.obs_means[state],
                cov=self.obs_covs[state],
                size=self.mc_samples
            )
        else:
            points = np.array([self.obs_means[state]]).reshape(1, -1)

        isAnd = True
        # Extract antecedents from the rule
        if "AND" not in rule:
            antecedents = rule.split("IF")[1].split("THEN")[0].strip().split("OR")
            isAnd = False
        else:
            antecedents = rule.split("IF")[1].split("THEN")[0].strip().split("AND")
        antecedents = [antecedent.strip() for antecedent in antecedents]

        # Initialize match score
        match_scores = np.ones(points.shape[0]) if isAnd else np.zeros(points.shape[0])

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
            if variable in self.obs_var_index:
                idx = self.obs_var_index[variable]
                vals = points[:, idx]
                membership_degree = np.array([fuzzy_set.get_value(v) for v in vals])
            elif action is not None:
                membership_degree = fuzzy_set.get_value(action)
            else:
                raise ValueError(f"Variable {variable} not recognized in rule matching.")

            # Update the match score (you can use different aggregation methods)
            if isAnd:
                match_scores = match_scores * membership_degree
            else:
                match_scores = _prob_sum(match_scores, membership_degree)

        return np.mean(match_scores)

    def _match_rule_cons(self, rule, action, O_means, state=0):
        """
        Check how much the consequent of the rule matches the current observation distribution given a state.
        """
        # Extract consequents from the rule
        consequent = rule.split("THEN")[1].strip()

        if bool(self.fuzzy_model._outputfunctions):
            return self._match_rule_cons_fun(rule, consequent, action, O_means, state)
        else:
            return self._match_rule_cons_mf(consequent, O_means, state)

    def _match_rule_cons_mf(self, consequent, O_means, state=0):
        """
        Return the consequent value given the observation model.
        In this case the consequent is modelled as a membership function
        """
        if self.integration_method == "montecarlo":
            points = np.random.multivariate_normal(
                mean=self.obs_means[state],
                cov=self.obs_covs[state],
                size=self.mc_samples
            )
        else:
            points = np.array([self.obs_means[state]]).reshape(1, -1)

        consequent = consequent.replace("(", " ")
        consequent = consequent.replace(")", " ")
        variable, term = consequent.split("IS")
        variable = variable.strip()
        term = term.strip()
        mapping_value = self.obs_var_index[variable]

        # Get the fuzzy set for the variable and term
        fuzzy_set = self.fuzzy_model.get_fuzzy_set(variable, term)
        membership_degree = [fuzzy_set.get_value(p[mapping_value]) for p in points]

        # Use a safer eval approach
        o_rule_pred = np.mean(membership_degree)
        return o_rule_pred, variable

    def _match_rule_cons_fun(self, rule, consequent, action, O_means, state=0):
        """
        Return the consequent value given the observation model.
        In this case the consequent is modelled as a (linear) function.
        """
        #TODO: generalize for multiple outputs
        #Once is generalized we can remove the rule parameter.
        if 'next_test' in rule or 'next_symptoms' in rule:
            function_str = consequent.split("IS")[1][1:5]
            term = consequent.split("IS")[0].replace("(", "").strip()
            fun = self.fuzzy_model._outputfunctions[function_str]

            for var in self.obs_var_index.keys():
                if var in fun:
                    idx = self.obs_var_index[var]
                    fun = fun.replace(var, str(O_means[state][idx]))

            if 'action' in fun:
                fun = fun.replace('action', str(action))

            # Use a safer eval approach
            o_rule_pred = eval(fun)
            return o_rule_pred, term

    def _marginal_likelihood(self, observation, state, obs_idx):
        """Compute the marginal likelihood of an observation given a state"""
        marginal_likelihood = norm.pdf(
            observation,
            loc=self.obs_means[state][obs_idx],
            scale=np.sqrt(np.diag(self.obs_covs[state])[obs_idx])
        )
        return marginal_likelihood

    def _compute_fuzzy_pseudo_counts(self):
        if not self.use_fuzzy:
            return None, None, None, None

        rules = self.fuzzy_model.get_rules()

        #Initialize pseudo-counts
        pseudo_count_T = np.zeros((self.n_states, self.n_actions, self.n_states))

        pseudo_count_O_den = np.zeros(self.n_states)
        pseudo_count_O_mean = np.zeros((self.n_states, self.obs_dim))
        pseudo_count_O_cov = np.zeros((self.n_states, self.obs_dim, self.obs_dim))

        for s in range(self.n_states):
            for a in range(self.n_actions):

                for rule in rules:

                    match_score = self._match_rule_ant(rule, a, s)
                    cons_s, var_name = self._match_rule_cons(rule, a, self.obs_means, s)

                    if match_score < 1e-9: continue

                    clean_var_name = var_name.replace("next_", "").strip()
                    idx = self.obs_var_index.get(clean_var_name)

                    if idx is None: continue

                    for s_prime in range(self.n_states):
                        strength = match_score * self.transitions[s, a, s_prime]
                        pseudo_count_O_den[s_prime] += strength

                        mean_copy = np.copy(self.obs_means[s])
                        mean_copy[idx] = cons_s
                        pdf_s_prime = self._marginal_likelihood(cons_s, s_prime, idx)
                        pseudo_count_O_mean[s_prime, :] += strength * mean_copy * pdf_s_prime
                        pseudo_count_T[s, a, s_prime] += match_score * pdf_s_prime

                        pseudo_count_O_cov[s_prime, :, :] += strength * np.outer(mean_copy, mean_copy)

        return pseudo_count_T, pseudo_count_O_den, pseudo_count_O_mean, pseudo_count_O_cov

    def maximization_step(self, observations, actions, gammas, xis, iteration=0):
        """M-step: Update model parameters based on expected sufficient statistics"""
        # Number of sequences
        n_observations = len(observations)

        emp_N_T = np.zeros_like(self.transitions)
        emp_den_T = np.zeros((self.n_states, self.n_actions))

        emp_N_O = np.zeros(self.n_states)
        emp_Sum_O = np.zeros((self.n_states, self.obs_dim))
        emp_Sum_sq_O = np.zeros((self.n_states, self.obs_dim, self.obs_dim))

        for i in range(n_observations):
            obs_np = np.array(observations[i])
            actions_np = np.array(actions[i])
            for a in range(self.n_actions):

                t_indices = np.where(actions_np[:-1] == a)[0]
                if len(t_indices) > 0:
                    emp_N_T[:, a, :] += np.sum(xis[i][a], axis=0)
                    emp_den_T[:, a] += np.sum(xis[i][a], axis=(0, 1))

            for s in range(self.n_states):
                gamma_s = gammas[i][:, s]
                emp_N_O[s] += np.sum(gamma_s)
                emp_Sum_O[s] += gamma_s @ obs_np
                emp_Sum_sq_O[s] += (obs_np.T * gamma_s) @ obs_np

        if self.use_fuzzy:
            fuzzy_N_T, fuzzy_N_O, fuzzy_Sum_O, fuzzy_Sum_Sq_O = self._compute_fuzzy_pseudo_counts()
        else:
            fuzzy_N_T = np.zeros_like(self.transitions)
            fuzzy_N_O = np.zeros(self.n_states)
            fuzzy_Sum_O = np.zeros((self.n_states, self.obs_dim))
            fuzzy_Sum_Sq_O = np.zeros((self.n_states, self.obs_dim, self.obs_dim))

        # Hyperparameter update
        if iteration > 10:
            if self.use_fuzzy and self.hp_method == "adaptive":
                self._update_hyperparameters_adaptive(fuzzy_N_T, fuzzy_N_O, len(observations))
            elif self.use_fuzzy and self.hp_method == "empirical_bayes":
                self._empirical_bayes_transition(emp_N_T, fuzzy_N_T)
                self._empirical_bayes_observation(emp_N_O, emp_Sum_O, emp_Sum_sq_O,
                                                  fuzzy_N_O, fuzzy_Sum_O, fuzzy_Sum_Sq_O)

        # Transitions
        if self.fix_transitions is None:
            lambda_T_reshaped = self.lambda_T[:, None, None]

            num = emp_N_T + fuzzy_N_T * lambda_T_reshaped
            den = np.sum(num, axis=2, keepdims=True)
            den[den == 0] = 1.0  # To avoid division by zero
            self.transitions = num / den

        # Observations
        for s in range(self.n_states):
            if s in self.fixed_observation_states:
                continue
            if self.fix_observations is None:
                den_O = emp_N_O[s] + self.lambda_O[s] * fuzzy_N_O[s] + self.epsilon_prior
                mean_O = (emp_Sum_O[s] + self.lambda_O[s] * fuzzy_Sum_O[s] + self.epsilon_prior) / den_O
                self.obs_means[s] = mean_O

                if self.use_fuzzy and self.hp_method != "empirical_bayes":
                    cov_O = (emp_Sum_sq_O[s] + self.lambda_O[s] * fuzzy_Sum_Sq_O[s]) / den_O - np.outer(mean_O, mean_O)
                else:
                    cov_O = self._compute_cov_empirical_bayes(emp_N_O, emp_Sum_O, emp_Sum_sq_O,
                                                              fuzzy_N_O, fuzzy_Sum_O, fuzzy_Sum_Sq_O, s)

                # Ensure covariance matrix is positive definite
                cov_O += self.epsilon_prior * np.eye(self.obs_dim)

                self.obs_covs[s] = cov_O

        self.initial_prob = np.mean([g[0] for g in gammas], axis=0)
        return

    def fit(self, observations, actions, max_iterations=100, tolerance=1e-12):
        """Run EM algorithm to fit the POMDP model parameters"""
        prev_ll = -np.inf

        history = {
            "log_likelihood": [],
            "lambda_T": [],
            "lambda_O": []
        }

        history["lambda_T"].append(self.lambda_T.copy())
        history["lambda_O"].append(self.lambda_O.copy())
        try:
            for iteration in range(max_iterations):
                # E-step
                if self.parallel:
                    gammas, xis, log_likelihood = self.expectation_step_parallel(observations, actions)
                else:
                    gammas, xis, log_likelihood = self.expectation_step(observations, actions)

                # Check convergence
                history["log_likelihood"].append(log_likelihood)
                improvement = log_likelihood - prev_ll
                self.improvement = improvement
                prev_ll = log_likelihood

                # M-step
                self.maximization_step(observations, actions, gammas, xis, iteration)

                history["lambda_T"].append(self.lambda_T.copy())
                history["lambda_O"].append(self.lambda_O.copy())

                if self.verbose:
                    print(
                        f"Iteration {iteration + 1},"
                        f" Log-Likelihood: {log_likelihood:.4f},"
                        f" Improvement: {improvement:.5f},"
                        f" Lambda_T: {np.array2string(history['lambda_T'][iteration], precision=4, separator=', ')},"
                        f" Lambda_O: {np.array2string(history['lambda_O'][iteration], precision=4, separator=', ')}")

                if np.abs(improvement) < tolerance:
                    print(f"Converged after {iteration + 1} iterations")
                    if self.verbose:
                        print(
                            f"Gamma: {np.array2string(gammas[0], precision=4, separator=', ')},")
                    break

        except ValueError as e:
            print(f"An error occurred during EM fitting: {e}")
            raise Exception(e)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            raise Exception(e)

        return log_likelihood

    def _update_hyperparameters_adaptive(self, curr_T_counts, curr_O_counts, n_sequences):
        VOLATILITY_THRESHOLD = 0

        if self.improvement < 0:
            if self.prev_fuzzy_counts_T is None or self.prev_fuzzy_counts_O is None:
                self.prev_fuzzy_counts_T = curr_T_counts
                self.prev_fuzzy_counts_O = curr_O_counts
                return
            else:
                for s in range(self.n_states):
                    # Transitions
                    delta_T = np.abs(curr_T_counts[s] - self.prev_fuzzy_counts_T[s])
                    D_s_T = np.mean(delta_T)
                    sigma_data = np.trace(self.obs_covs[s]) + 1e-6
                    normalized_volatility_T = (D_s_T / sigma_data)


                    decay_T = 1.0 / (1.0 + self.alpha_ah * normalized_volatility_T)
                    # Observations
                    # We use the fuzzy_O denominators as counts
                    delta_O = np.abs(curr_O_counts[s] - self.prev_fuzzy_counts_O[s])
                    D_s_O = np.mean(delta_O)

                    normalized_volatility_O = (D_s_O / sigma_data)
                    decay_O = 1.0 / (1.0 + self.alpha_ah * normalized_volatility_O)
                    if normalized_volatility_T > VOLATILITY_THRESHOLD:
                        self.lambda_T[s] = max(self.lambda_T[s] * decay_T, self.lambda_min)
                    if normalized_volatility_O > VOLATILITY_THRESHOLD:
                        self.lambda_O[s] = max(self.lambda_O[s] * decay_O, self.lambda_min)
                    if self.verbose:
                        print(
                            f"Adaptive Update: Lambda_T: {np.array2string(self.lambda_T, precision=4, separator=', ')},"
                            f" Lambda_O:{np.array2string(self.lambda_O, precision=8, separator=', ')}")

            self.prev_fuzzy_counts_T = curr_T_counts
            self.prev_fuzzy_counts_O = curr_O_counts

    def _empirical_bayes_transition(self, emp_N_T, fuzzy_N_T):
        """Empirical Bayes adaptation of lambda for transitions"""
        grad_lambda_T = np.zeros(self.n_states)

        for s in range(self.n_states):
            grad_sum_s = 0.0
            for a in range(self.n_actions):
                n_emp = emp_N_T[s, a, :]
                n_fuzzy = fuzzy_N_T[s, a, :]

                alpha_prior = self.lambda_T[s] * n_fuzzy + 1
                beta_post = n_emp + alpha_prior

                sum_alpha = np.sum(alpha_prior)
                sum_beta = np.sum(beta_post)

                #Gradient computation
                term1 = np.sum(n_fuzzy * (digamma(beta_post) - digamma(alpha_prior)))
                term2 = np.sum(n_fuzzy) * (digamma(sum_beta) - digamma(sum_alpha))

                grad_sum_s += (term1 - term2)
            grad_lambda_T[s] = grad_sum_s

        # Apply gradient ascent step
        self.lambda_T += self.eb_lr * grad_lambda_T
        self.lambda_T = np.maximum(self.lambda_T, 0)

    def _empirical_bayes_observation(self, emp_N_O, emp_Sum_O, emp_Sum_sq_O,
                                     fuzzy_N_O, fuzzy_Sum_O, fuzzy_Sum_Sq_O):
        """Empirical Bayes adaptation of lambda for observations"""
        grad_lambda_O = np.zeros(self.n_states)

        for s in range(self.n_states):
            n_emp = emp_N_O[s]
            sum_emp = emp_Sum_O[s]
            sum_sq_emp = emp_Sum_sq_O[s]

            n_fuzzy = fuzzy_N_O[s]
            sum_fuzzy = fuzzy_Sum_O[s]
            sum_sq_fuzzy = fuzzy_Sum_Sq_O[s]

            #Parameters of the Normal-Inverse-Wishart prior
            mu_0 = sum_fuzzy / (n_fuzzy + self.epsilon_prior)
            kappa_0 = self.lambda_O[s] * n_fuzzy + self.epsilon_prior
            nu_0 = self.lambda_O[s] * n_fuzzy + self.obs_dim
            psi_fuzzy = sum_sq_fuzzy - n_fuzzy * np.outer(mu_0, mu_0)
            psi_0 = psi_fuzzy * self.lambda_O[s] + self.epsilon_prior * np.eye(self.obs_dim)

            mu_data = sum_emp / (n_emp + self.epsilon_prior)
            psi_data = sum_sq_emp - (np.outer(sum_emp, sum_emp) / n_emp)
            diff_mu_data_0 = mu_data - mu_0

            # Parameters of the Normal-Inverse-Wishart posterior
            kappa_n = kappa_0 + n_emp
            nu_n = nu_0 + n_emp
            mu_n = (kappa_0 * mu_0 + n_emp * mu_data) / (kappa_0 + n_emp)
            psi_n = psi_0 + psi_data + ((kappa_0 * n_emp) / (kappa_0 + n_emp)) * np.outer(diff_mu_data_0, diff_mu_data_0)

            # Gradient computation

            # Precompute determinants and inverse
            det_psi_0 = np.linalg.det(psi_0)
            det_psi_n = np.linalg.det(psi_n)
            ln_det_ratio = np.log(det_psi_0 + self.epsilon_prior) - np.log(det_psi_n + self.epsilon_prior)

            inv_psi_n = np.linalg.inv(psi_n + self.epsilon_prior * np.eye(self.obs_dim))

            #
            term_1 = n_fuzzy / 2 * (
                    multidigamma(nu_n / 2, self.obs_dim) -
                    multidigamma(nu_0 / 2, self.obs_dim) +
                    ln_det_ratio
            )

            term_2 = (self.obs_dim * n_emp * n_fuzzy) / (2 * kappa_0 * kappa_n)

            term_3 = (nu_0 * self.obs_dim) / (2 * self.lambda_O[s] + + self.epsilon_prior)

            diff_mu_mul = np.outer(diff_mu_data_0, diff_mu_data_0)
            scalar_factor = ((n_emp ** 2 * n_fuzzy) / kappa_n ** 2)
            matrix_term = (psi_fuzzy + scalar_factor * diff_mu_mul)
            term_4 = - (nu_n / 2) * np.trace(inv_psi_n @ matrix_term)

            grad_lambda_O[s] = term_1 + term_2 + term_3 + term_4

        # Apply gradient ascent step
        self.lambda_O += self.eb_lr * 1/10 * grad_lambda_O
        self.lambda_O = np.maximum(self.lambda_O, 0)

    def _compute_cov_empirical_bayes(self, emp_N_O, emp_Sum_O, emp_Sum_sq_O,
                                     fuzzy_N_O, fuzzy_Sum_O, fuzzy_Sum_Sq_O, state):
        """Compute covariance matrices for Empirical Bayes"""
        n_emp = emp_N_O[state]
        sum_emp = emp_Sum_O[state]
        sum_sq_emp = emp_Sum_sq_O[state]

        n_fuzzy = fuzzy_N_O[state]
        sum_fuzzy = fuzzy_Sum_O[state]
        sum_sq_fuzzy = fuzzy_Sum_Sq_O[state]

        mu_0 = sum_fuzzy / (n_fuzzy + self.epsilon_prior)
        kappa_0 = self.lambda_O[state] * n_fuzzy + self.epsilon_prior
        nu_0 = self.lambda_O[state] * n_fuzzy + self.obs_dim
        psi_fuzzy = sum_sq_fuzzy - n_fuzzy * np.outer(mu_0, mu_0)
        psi_0 = psi_fuzzy * self.lambda_O[state] + self.epsilon_prior * np.eye(self.obs_dim)

        mu_data = sum_emp / (n_emp + self.epsilon_prior)
        psi_data = sum_sq_emp - (np.outer(sum_emp, sum_emp) / n_emp)
        diff_mu_data_0 = mu_data - mu_0

        nu_n = nu_0 + n_emp
        psi_n = psi_0 + psi_data + ((kappa_0 * n_emp) / (kappa_0 + n_emp)) * np.outer(diff_mu_data_0, diff_mu_data_0)

        cov_matr = psi_n / (nu_n + self.obs_dim + 1)
        return cov_matr


#TODO: move the below functions to a different file
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

    for lambda_T in rho_values:
        for lambda_O in rho_values:
            print(f"\nRunning POMDP reconstruction with lambda_T={lambda_T} and lambda_O={lambda_O}...")
            pomdp_model = FuzzyPOMDP(
                n_states=n_states,
                n_actions=n_actions,
                obs_dim=obs_dim,
                use_fuzzy=True,
                lambda_T=lambda_T,
                lambda_O=lambda_O,
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

            str_rho = f"lambda_T={lambda_T}, lambda_O={lambda_O}"
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
        results: Dictionary with keys 'lambda_T=X, lambda_O=Y' and distance values
    """
    # Extract unique rho values
    lambda_T_values = []
    lambda_O_values = []

    for key in results.keys():
        parts = key.split(',')
        lambda_T = float(parts[0].split('=')[1])
        lambda_O = float(parts[1].split('=')[1])

        if lambda_T not in lambda_T_values:
            lambda_T_values.append(lambda_T)
        if lambda_O not in lambda_O_values:
            lambda_O_values.append(lambda_O)

    # Sort the values
    lambda_T_values.sort()
    lambda_O_values.sort()

    # Create the heatmap data
    heatmap_data = np.zeros((len(lambda_T_values), len(lambda_O_values)))

    for i, lambda_T in enumerate(lambda_T_values):
        for j, lambda_O in enumerate(lambda_O_values):
            key = f"lambda_T={lambda_T}, lambda_O={lambda_O}"
            heatmap_data[i, j] = results[key]

    # Create the plot
    plt.figure(figsize=(10, 8))
    im = plt.imshow(heatmap_data, cmap='viridis_r')  # viridis_r makes lower values (better) darker

    # Add colorbar
    cbar = plt.colorbar(im)
    #cbar.set_label('L1 Distance (lower is better)')

    # Set labels
    plt.title('POMDP Reconstruction Sensitivity Analysis')
    plt.xlabel('lambda_O values')
    plt.ylabel('lambda_T values')

    # Set ticks
    plt.xticks(np.arange(len(lambda_O_values)), [f"{x:.2f}" for x in lambda_O_values])
    plt.yticks(np.arange(len(lambda_T_values)), [f"{x:.2f}" for x in lambda_T_values])

    # Add text annotations
    for i in range(len(lambda_T_values)):
        for j in range(len(lambda_O_values)):
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
        lambda_T=0.0,
        lambda_O=0.0,
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


def _visualize_comparison_observation_distributions(models, n_states, title_prefix="", datasize="N/A"):
    """Visualize and compare the observation distributions for each state for each model."""
    n_models = len(models)
    rows = n_models
    cols = n_states
    plt.figure(figsize=(5 * cols, 4 * rows))
    plt.suptitle(f"Datasize {datasize}", fontsize=16)

    # Create a single grid once
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)

    for m, model in enumerate(models):
        pomdp_model = model['model']
        perm_order = model['perm_ord']
        name = model['name']

        pomdp_model.obs_means = pomdp_model.obs_means[perm_order]
        pomdp_model.obs_covs = pomdp_model.obs_covs[perm_order]
        for s in range(n_states):
            ax = plt.subplot(rows, cols, m * cols + s + 1)

            # Compute the PDF on the grid for the current model and state
            Z = np.zeros_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    obs = np.array([X[i, j], Y[i, j]])
                    Z[i, j] = pomdp_model.observation_likelihood(obs, s)

            cf = ax.contourf(X, Y, Z, levels=20, cmap='viridis')
            ax.set_title(f"{title_prefix} Model {name} State {s}")
            ax.set_xlabel("Test Result")
            ax.set_ylabel("Symptoms")
            plt.colorbar(cf, ax=ax)

    plt.tight_layout()
    plt.show()


def visualize_observation_distributions(model, n_states, title_prefix="", datasize="N/A"):
    """Visualize the observation distributions for each state"""
    plt.figure(figsize=(15, 5))
    plt.suptitle(f"Datasize {datasize}", fontsize=16)
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
    plt.show()


def run_pomdp_reconstruction():
    """Main experiment function for POMDP reconstruction"""
    # Parameters
    n_trajectories = 25
    trajectory_length = 5
    n_states = 3  # healthy, sick, critical
    n_actions = 2  # wait, treat
    obs_dim = 2  # test_result and symptoms (continuous)
    seed = 15  # For reproducibility

    # Generate training data
    original_pomdp, observations, actions, true_states, rewards = generate_pomdp_data(
        n_trajectories, trajectory_length, seed=seed
    )

    fuzzy_model = build_fuzzymodel()

    # Train fuzzy POMDP model
    fuzzy_pomdp = FuzzyPOMDP(
        n_states=n_states,
        n_actions=n_actions,
        obs_dim=obs_dim,
        use_fuzzy=True,
        fuzzy_model=fuzzy_model,
        lambda_T=0.1,
        lambda_O=0.0,
        verbose=False
        # fix_transitions=np.array(DEFAULT_PARAMS_TRANS),  # Set to None to allow learning transitions
    )

    fuzzy_ll = fuzzy_pomdp.fit(
        observations, actions,
        max_iterations=200, tolerance=1e-8
    )

    print(f"Fuzzy POMDP transitions:\n{fuzzy_pomdp.transitions}")

    # Visualize fuzzy model results
    visualize_observation_distributions(fuzzy_pomdp, n_states, title_prefix="Fuzzy")
    plt.show()

    # Compare with original model
    fuzzy_pomdp.compare_state_transitions(
        fuzzy_pomdp.transitions,
        original_pomdp.env.transition_model.transitions,
        original_pomdp.agent.observation_model,
        STATES
    )

    # Train standard POMDP model
    standard_pomdp, standard_ll = train_pomdp_model(
        observations, actions, use_fuzzy=False,
        n_states=n_states, n_actions=n_actions, obs_dim=obs_dim,
        max_iterations=200, tolerance=1e-8
    )

    print(f"Standard POMDP transitions:\n{standard_pomdp.transitions}")

    # Visualize standard model results
    visualize_observation_distributions(standard_pomdp, n_states, title_prefix="Standard")
    plt.show()

    # Compare with original observation distribution
    #print("\nComparing with original observation distributions...")
    #plot_observation_distribution()

    # Compare standard model with original model
    print("Evaluating standard model against original model...")
    standard_pomdp.compare_state_transitions(
        standard_pomdp.transitions,
        original_pomdp.env.transition_model.transitions,
        original_pomdp.agent.observation_model,
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



if __name__ == "__main__":
    learned_pomdp = main()
    #evaluate_fuzzy_reward_prediction(trials=200, horizon=10)
    #sensitivity_results = rho_sensitivity_analysis()
