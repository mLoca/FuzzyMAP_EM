import random
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans

from sklearn.mixture import GaussianMixture
import multiprocessing as mp


#TODO: add a parameter for the regularization of the covariance matrices
class PomdpEM:
    """
    EM algorithm implementation for POMDPs with:
    - Discrete states
    - Discrete actions
    - Continuous observations (modeled with fuzzy logic)
    """

    def __init__(self, n_states: int, n_actions: int, obs_dim: int, verbose: bool = False, seed: int = 42,
                 parallel: bool = False, epsilon_prior=1e-4, ensure_psd=False):
        """Initialize the POMDP model with EM capabilities"""
        self.n_states = n_states
        self.n_actions = n_actions
        self.obs_dim = obs_dim

        # Initialize model parameters
        self.transitions = np.ones((n_states, n_actions, n_states)) / n_states  # P(s'|s,a)
        self.initial_prob = np.ones(n_states) / n_states  # P(s_0)

        # Observation model parameters (Gaussian for each state)
        self.obs_means = np.random.normal(loc=np.abs(np.random.randn(1)), scale=0.1, size=(n_states, obs_dim))
        self.obs_covs = np.array([np.eye(obs_dim) for _ in range(n_states)])

        self.parallel = parallel
        self.epsilon_prior = epsilon_prior
        self.ensure_psd = ensure_psd
        self.verbose = verbose

        np.random.seed(seed)
        random.seed(seed)

    def initialize_with_kmeans(self, observations):
        """
        Initializes observation model means using K-Means clustering.
        """
        all_obs = np.vstack([obs_seq for obs_seq in observations])

        kmeans = KMeans(n_clusters=self.n_states, random_state=0)
        kmeans.fit(all_obs)

        # Assign the cluster centers as the initial means for the observation model
        self.obs_means = kmeans.cluster_centers_
        if self.verbose:
            print("Initialized observation means with K-Means.")

    def observation_likelihood(self, obs, state):
        """Compute P(o|s) using multivariate Gaussian"""
        cov = self.obs_covs[state] if not self.ensure_psd else self._ensure_psd(self.obs_covs[state])

        return multivariate_normal.pdf(
            obs,
            mean=self.obs_means[state],
            cov=cov,
            allow_singular=True
        )

    def _compute_emission_matrix(self, observations):
        """
        Compute the emission probability matrix for all observations and states
        Emission_Matrix: Emission matrix (T, n_states)
        """
        T = len(observations)
        obs_probs = np.zeros((T, self.n_states))

        for s in range(self.n_states):
            cov = self._ensure_psd(self.obs_covs[s])

            obs_probs[:, s] = multivariate_normal.pdf(observations, mean=self.obs_means[s], cov=cov,
                                                      allow_singular=True)

        obs_probs[obs_probs < 1e-50] = 1e-51
        return obs_probs

    def forward_pass(self, obs_probs, actions):
        """Forward algorithm computing α_t(s)"""
        T = obs_probs.shape[0]
        alpha = np.zeros((T, self.n_states))
        c = np.zeros(T)

        # Initialize with alpha_0 = P(s_0) * P(o_0|s_0)
        alpha[0, :] = self.initial_prob * obs_probs[0, :]
        c[0] = np.sum(alpha[0, :])
        alpha[0, :] /= c[0] + 1e-20

        # Forward recursion
        for t in range(1, T):
            action = actions[t - 1]

            # Prob of arriving at s from any prev state
            alpha_pred = alpha[t - 1] @ self.transitions[:, action, :].squeeze()

            alpha[t, :] = alpha_pred * obs_probs[t, :]
            c[t] = np.sum(alpha[t, :])
            alpha[t, :] /= c[t] + 1e-20

        log_likelihood = np.sum(np.log(c))
        return alpha, log_likelihood, c

    def backward_pass(self, obs_prob, actions, c):
        """Backward algorithm computing β_t(s)"""
        T = obs_prob.shape[0]
        beta = np.zeros((T, self.n_states))

        # Initialize with beta_(T-1) = 1
        beta[T - 1] = np.ones(self.n_states)

        # Backward recursion
        for t in range(T - 2, -1, -1):
            action = actions[t]

            # prob of transitioning to s' AND seeing obs[t+1]
            term = obs_prob[t + 1] * beta[t + 1]
            beta[t] = self.transitions[:, action, :].squeeze() @ term

            beta[t] /= (np.sum(c[t + 1]) + 1e-20)

        return beta

    def compute_posterior(self, alpha, beta):
        """Compute γ_t(s) = P(s_t=s|o_1:T, a_1:T)"""
        gamma = alpha * beta
        gamma /= np.sum(gamma, axis=1, keepdims=True) + 1e-10
        return gamma

    def compute_transition_counts(self, alpha, beta, obs_prob, actions):
        """
        Compute xi (transition counts).
        Returns: xi of shape (T-1, n_states, n_states)
        """
        T = obs_prob.shape[0]
        xi = np.zeros((T - 1, self.n_states, self.n_states))

        for t in range(T - 1):
            action = actions[t]

            term = obs_prob[t + 1] * beta[t + 1].reshape(1, -1)
            xi[t] = (alpha[t].reshape(-1, 1) * self.transitions[:, action, :].squeeze()) * term
            xi[t] /= np.sum(xi[t]) + 1e-20

        return xi

    def _process_single_sequence(self, obs_seq, act_seq):
        """Process a single observation sequence"""
        obs_probs = self._compute_emission_matrix(obs_seq)

        alpha, seq_ll, c = self.forward_pass(obs_probs, act_seq)
        beta = self.backward_pass(obs_probs, act_seq, c)

        gamma = self.compute_posterior(alpha, beta)
        xi = self.compute_transition_counts(alpha, beta, obs_probs, act_seq)

        return gamma, xi, seq_ll

    def expectation_step(self, observations, actions):
        """E-step: Compute expected sufficient statistics"""
        gammas, xis = [], []
        total_ll = 0.0

        for obs_seq, act_seq in zip(observations, actions):
            gamma, xi, seq_ll = self._process_single_sequence(obs_seq, act_seq)
            gammas.append(gamma)
            xis.append(xi)
            total_ll += seq_ll
        return gammas, xis, total_ll

    def expectation_step_parallel(self, observations, actions):
        """Parallelized E-step using multi-processing"""
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=min(mp.cpu_count(), 8)) as pool:
            results = pool.starmap(
                self._process_single_sequence,
                [(np.array(obs_seq), np.array(act_seq)) for obs_seq, act_seq in zip(observations, actions)]
            )

        # Unpack results
        gammas, xis, seq_lls = zip(*results)
        total_ll = sum(seq_lls)

        return list(gammas), list(xis), total_ll

    def maximization_step(self, observations, actions, gammas, xis, iteration=0):
        """M-step: Update model parameters based on expected sufficient statistics"""
        n_sequences = len(observations)

        # Update initial state probabilities
        self.initial_prob = np.zeros(self.n_states)
        for i in range(n_sequences):
            self.initial_prob += gammas[i][0]
        self.initial_prob /= n_sequences

        # Update transition probabilities
        for s in range(self.n_states):
            for a in range(self.n_actions):
                numerator = np.zeros(self.n_states)
                denominator = 0.0

                for i in range(n_sequences):
                    for t in range(len(actions[i]) - 1):
                        if actions[i][t] == a:
                            # Add counts for state-action transitions
                            for s_prime in range(self.n_states):
                                numerator[s_prime] += xis[i][t, s, s_prime]
                            denominator += gammas[i][t, s]

                # Update transition probabilities if we have observations
                if denominator > 0:
                    self.transitions[s, a, :] = numerator / denominator

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
        for s in range(self.n_states):
            if obs_counts[s] > 0:
                total_weight = obs_counts[s] + self.epsilon_prior
                self.obs_means[s] = new_means[s] / total_weight

                # Update covariance
                E_ooT_s = data_O_sum_gamma_obs_sq[s] / total_weight
                mu_s_outer_mu_s = np.outer(self.obs_means[s, :], self.obs_means[s, :])
                self.obs_covs[s, :, :] = E_ooT_s - mu_s_outer_mu_s
                if self.ensure_psd:
                    self.obs_covs[s, :, :] = self._ensure_psd(self.obs_covs[s, :, :])
                else:
                    self.obs_covs[s, :, :] += self.epsilon_prior * np.eye(self.obs_dim)  # Regularization

        return

    def _ensure_psd(self, sigma, min_val=None):
        """Ensure covariance matrix is positive semi-definite"""
        if min_val is None:
            min_val = self.epsilon_prior

        sigma = 0.5 * (sigma + sigma.T)
        return sigma + np.eye(sigma.shape[0]) * min_val

    def fit(self, observations, actions, max_iterations=100, tolerance=1e-4):
        """Run EM algorithm to fit the POMDP model parameters"""
        prev_ll = -np.inf
        log_likelihood = None
        try:
            for iteration in range(max_iterations):
                # E-step
                if self.parallel:
                    gammas, xis, log_likelihood = self.expectation_step_parallel(observations, actions)
                else:
                    gammas, xis, log_likelihood = self.expectation_step(observations, actions)

                # Check convergence
                improvement = log_likelihood - prev_ll
                prev_ll = log_likelihood

                if self.verbose:
                    print(
                        f"Iteration {iteration + 1}, Log-Likelihood: {log_likelihood:.4f}, Improvement: {improvement:.4f}")

                if np.abs(improvement) < tolerance:
                    print(f"Converged after {iteration + 1} iterations")
                    break

                # M-step
                self.maximization_step(observations, actions, gammas, xis)

        except ValueError as e:
            print(f"An error occurred during EM fitting: {e}")
            raise Exception(e)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            raise Exception(e)

        return log_likelihood
