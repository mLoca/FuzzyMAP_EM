import numpy as np
from scipy.special import digamma
from scipy.stats import norm

from models.trainable.pomdp_EM import PomdpEM

from fuzzy.fuzzy_model import build_fuzzymodel
from utils.utils import _prob_sum, multidigamma


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
                 ensure_psd=False,
                 integration_method="mean",  # 'mean' or 'montecarlo'
                 mc_samples=100,
                 hyperparameter_update_method="static",  # 'static', 'adaptive', 'empirical_bayes'
                 eb_learning_rate_O=0.01,
                 eb_learning_rate_T=0.01,
                 warm_start_update=10,
                 alpha_ah=0.05,
                 lambda_min=0.0,
                 epsilon_prior=1e-4
                 ):
        super().__init__(n_states, n_actions, obs_dim, verbose, parallel=parallel, seed=seed,
                         epsilon_prior=epsilon_prior, ensure_psd=ensure_psd)

        self.transition_inertia = 0
        """Initialize the POMDP model with EM capabilities"""

        self.use_fuzzy = use_fuzzy
        self.lambda_T = np.ones(self.n_states) * lambda_T  # weight parameter for pseudo-counts
        self.lambda_O = np.ones(self.n_states) * lambda_O  # weight parameter for pseudo-counts in observation model
        self.fix_transitions = fix_transitions
        self.fix_observations = fix_observations
        self.fixed_observation_states = fixed_observation_states if fixed_observation_states is not None else []

        if fix_transitions is not None:
            self.transitions = np.array(fix_transitions, dtype=np.float64)

        # Integration Strategy
        self.integration_method = integration_method  # 'mean' or 'monte_carlo'
        self.mc_samples = mc_samples

        # Hyperparameters for Empirical Bayes
        self.hp_method = hyperparameter_update_method
        self.eb_lr_T = eb_learning_rate_T
        self.eb_lr_O = eb_learning_rate_O

        #Adaptive Learning
        self.alpha_ah = alpha_ah
        self.warm_start_update = warm_start_update
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

        match_scores = np.ones(points.shape[0]) if isAnd else np.zeros(points.shape[0])

        # compute the match score for each antecedent
        for antecedent in antecedents:
            antecedent = antecedent.replace("(", " ")
            antecedent = antecedent.replace(")", " ")
            variable, term = antecedent.split("IS")
            variable = variable.strip()
            term = term.strip()

            # get the fuzzy set
            fuzzy_set = self.fuzzy_model.get_fuzzy_set(variable, term)

            # compute the membership degree
            if variable in self.obs_var_index:
                idx = self.obs_var_index[variable]
                vals = points[:, idx]
                membership_degree = np.array([fuzzy_set.get_value(v) for v in vals])
            elif action is not None:
                membership_degree = fuzzy_set.get_value(action)
            else:
                raise ValueError(f"Variable {variable} not recognized in rule matching.")

            # update the match scores depending on AND/OR
            if isAnd:
                match_scores = match_scores * membership_degree
            else:
                match_scores = _prob_sum(match_scores, membership_degree)

        return np.mean(match_scores)

    def _match_rule_cons(self, rule, action, state=0):
        """
        Check how much the consequent of the rule matches the current observation distribution given a state.
        """
        # Extract consequent from the rule
        consequent = rule.split("THEN")[1].strip()

        if bool(self.fuzzy_model._outputfunctions):
            return self._match_rule_cons_fun(rule, consequent, action, state)
        else:
            return self._match_rule_cons_mf(consequent, state)

    def _match_rule_cons_mf(self, consequent, state=0):
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

        o_rule_pred = np.mean(membership_degree)
        return o_rule_pred, variable

    def _match_rule_cons_fun(self, rule, consequent, action, state=0):
        """
        Return the consequent value given the observation model.
        In this case the consequent is modelled as a (linear) function.
        """
        if 'next_test' in rule or 'next_symptoms' in rule:

            function_str = consequent.split("IS")[1][1:5]
            term = consequent.split("IS")[0].replace("(", "").strip()
            clean_term = term.replace("next_", "").strip()
            fun = self.fuzzy_model._outputfunctions[function_str]

            for var in self.obs_var_index.keys():
                if var in fun:
                    idx = self.obs_var_index[var]
                    fun = fun.replace(var, str(self.obs_means[state][idx]))

            if 'action' in fun:
                fun = fun.replace('action', str(action))

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
                    cons_s, var_name = self._match_rule_cons(rule, a, s)

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

                        pseudo_count_O_mean[s_prime, :] += strength * mean_copy
                        pseudo_count_T[s, a, s_prime] += match_score * pdf_s_prime
                        pseudo_count_O_cov[s_prime, :, :] += strength * np.outer(mean_copy, mean_copy)

        return pseudo_count_T, pseudo_count_O_den, pseudo_count_O_mean, pseudo_count_O_cov

    def maximization_step(self, observations, actions, gammas, xis, iteration=0):
        """M-step: Update model parameters based on expected sufficient statistics"""
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
        if iteration > self.warm_start_update:
            if self.use_fuzzy and self.hp_method == "adaptive":
                self._update_hyperparameters_adaptive(fuzzy_N_T, fuzzy_N_O, len(observations))
            elif self.use_fuzzy and self.hp_method == "empirical_bayes":
                self._empirical_bayes_transition(emp_N_T, fuzzy_N_T)
                self._empirical_bayes_observation(emp_N_O, emp_Sum_O, emp_Sum_sq_O,
                                                  fuzzy_N_O, fuzzy_Sum_O, fuzzy_Sum_Sq_O)

        # Transitions
        if self.fix_transitions is None:
            lambda_T_reshaped = self.lambda_T[:, None, None]
            # create an inertia matrix to stabilize transition probabilities
            inertia_matrix = np.eye(self.n_states) * self.transition_inertia
            inertia_tensor = np.repeat(inertia_matrix[:, np.newaxis, :], self.n_actions, axis=1)

            num = emp_N_T + fuzzy_N_T * lambda_T_reshaped + inertia_tensor
            den = np.sum(num, axis=2, keepdims=True)
            den[den == 0] = 1.0  # Prevent division by zero
            self.transitions = num / den

        # Observations
        for s in range(self.n_states):
            if s in self.fixed_observation_states:
                continue
            if self.fix_observations is None:
                den_O = emp_N_O[s] + self.lambda_O[s] * fuzzy_N_O[s] + self.epsilon_prior
                mean_O = (emp_Sum_O[s] + self.lambda_O[s] * fuzzy_Sum_O[s]) / den_O
                self.obs_means[s] = mean_O

                if self.use_fuzzy and self.hp_method != "empirical_bayes":
                    cov_O = (emp_Sum_sq_O[s] + self.lambda_O[s] * fuzzy_Sum_Sq_O[s]) / den_O - np.outer(mean_O, mean_O)
                else:
                    cov_O = self._compute_cov_empirical_bayes(emp_N_O, emp_Sum_O, emp_Sum_sq_O,
                                                              fuzzy_N_O, fuzzy_Sum_O, fuzzy_Sum_Sq_O, s)

                # Ensure covariance matrix is positive definite
                if self.ensure_psd:
                    cov_O = self._ensure_psd(cov_O)
                else:
                    cov_O += self.epsilon_prior * np.eye(self.obs_dim)

                self.obs_covs[s] = cov_O

        self.initial_prob = np.mean([g[0] for g in gammas], axis=0)
        return

    def fit(self, observations, actions, max_iterations=100, tolerance=1e-12):
        """Run EM algorithm to fit the POMDP model parameters"""
        log_likelihood = None
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
                    sigma_data = np.trace(self.obs_covs[s]) + self.epsilon_prior
                    normalized_volatility_T = (D_s_T / sigma_data)

                    decay_T = 1.0 / (1.0 + self.alpha_ah * normalized_volatility_T)
                    # Observations
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
        self.lambda_T += self.eb_lr_T * grad_lambda_T
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
            psi_n = (psi_0 + psi_data + ((kappa_0 * n_emp) / (kappa_0 + n_emp)) * np.outer(diff_mu_data_0,
                                                                                           diff_mu_data_0)
                     + self.epsilon_prior * np.eye(self.obs_dim))

            # Compute determinants and inverse
            det_psi_0 = np.linalg.det(psi_0)
            det_psi_n = np.linalg.det(psi_n)
            ln_det_ratio = np.log(det_psi_0) - np.log(det_psi_n)
            inv_psi_n = np.linalg.inv(psi_n)

            # Gradient computation
            term_1 = n_fuzzy / 2 * (
                    multidigamma(nu_n / 2, self.obs_dim) -
                    multidigamma(nu_0 / 2, self.obs_dim) +
                    ln_det_ratio
            )

            term_2 = (self.obs_dim * n_emp * n_fuzzy) / (2 * kappa_0 * kappa_n)

            term_3 = (nu_0 * self.obs_dim) / (2 * self.lambda_O[s] + self.epsilon_prior)

            diff_mu_mul = np.outer(diff_mu_data_0, diff_mu_data_0)
            scalar_factor = ((n_emp ** 2 * n_fuzzy) / kappa_n ** 2)
            matrix_term = (psi_fuzzy + scalar_factor * diff_mu_mul)
            term_4 = - (nu_n / 2) * np.trace(inv_psi_n @ matrix_term)

            grad_lambda_O[s] = term_1 + term_2 + term_3 + term_4

        # Apply gradient ascent step
        self.lambda_O += self.eb_lr_O * 1 / 10 * grad_lambda_O
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
