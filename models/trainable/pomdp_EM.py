import random
from time import sleep

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans

from sklearn.mixture import GaussianMixture

from continouos_pomdp_example import plot_observation_distribution, STATES, generate_pomdp_data


#TODO: add a parameter for the regularization of the covariance matrices
class PomdpEM:
    """
    EM algorithm implementation for POMDPs with:
    - Discrete states
    - Discrete actions
    - Continuous observations (modeled with fuzzy logic)
    """

    def __init__(self, n_states: int, n_actions: int, obs_dim: int, verbose: bool = False, seed: int = 42,
                 parallel: bool = False, epsilon_prior = 1e-4,
                 ensure_psd = False):
        """Initialize the POMDP model with EM capabilities"""
        self.ensure_psd = ensure_psd
        self.n_states = n_states
        self.n_actions = n_actions
        self.obs_dim = obs_dim

        # Initialize model parameters
        self.transitions = np.ones((n_states, n_actions, n_states)) / n_states  # P(s'|s,a)
        self.initial_prob = np.ones(n_states) / n_states  # P(s_0)

        #Seed for reproducibility
        np.random.seed(seed)
        random.seed(seed)

        # Observation model parameters (Gaussian for each state)
        self.obs_means = np.random.normal(loc=np.abs(np.random.randn(1)), scale=0.1, size=(n_states, obs_dim))
        self.obs_covs = np.array([np.eye(obs_dim) for _ in range(n_states)])

        self.verbose = verbose
        self.parallel = parallel
        self.epsilon_prior = epsilon_prior

    def initialize_with_kmeans(self, observations):
        """
        Initializes observation model means using K-Means clustering.
        This provides a better starting point for the EM algorithm.
        """
        # Flatten the list of observation sequences into a single array
        all_obs = np.vstack([obs_seq for obs_seq in observations])

        # Use K-Means to find cluster centers
        kmeans = KMeans(n_clusters=self.n_states, random_state=0, n_init=40)
        kmeans.fit(all_obs)

        # Assign the cluster centers as the initial means for the observation model
        self.obs_means = kmeans.cluster_centers_
        if self.verbose:
            print("Initialized observation means with K-Means.")

    #TODO: remove it
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

        for t in range(T):
            for s in range(self.n_states):
                cov = self._ensure_psd(self.obs_covs[s])

                obs_probs[:, s] = multivariate_normal.pdf(observations, mean=self.obs_means[s], cov=cov,
                                                          allow_singular=True)
        obs_probs[obs_probs < 1e-300] = 1e-301  # Prevent exact zeros
        return obs_probs

    def forward_pass(self, obs_probs, actions):
        """Forward algorithm computing α_t(s)"""
        T = obs_probs.shape[0]
        alpha = np.zeros((T, self.n_states))
        c = np.zeros(T)

        # Initialize with alpha_0 = P(s_0) * P(o_0|s_0)
        alpha[0] = self.initial_prob * obs_probs[0]
        c[0] = np.sum(alpha[0])
        alpha[0] /= c[0] + 1e-20

        # Forward recursion
        for t in range(1, T):
            action = actions[t - 1]
            #Vectorized transition - alpha[t-1] (1xS) dot Trans(S, S') - (1xS')
            alpha_pred = alpha[t - 1] @ self.transitions[:, action, :].squeeze()

            alpha[t] = alpha_pred * obs_probs[t]
            c[t] = np.sum(alpha[t])
            alpha[t] /= c[t] + 1e-20  # Normalize to prevent

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

            term = obs_prob[t + 1] * beta[t + 1]
            #Vectorized transition - Trans(S, S') dot term (S') - (S,)
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

            term = obs_prob[t + 1] * beta[t + 1].reshape(1, -1)  # (1, S')
            #numeratore
            xi[t] = (alpha[t].reshape(-1, 1) * self.transitions[:, action, :].squeeze()) * term  # (S, S')
            # Normalize
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
        import multiprocessing as mp

        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=min(mp.cpu_count(), 8)) as pool:
            results = pool.starmap(
                self._process_single_sequence,
                [(obs_seq, act_seq) for obs_seq, act_seq in zip(observations, actions)]
            )

        # Unpack results
        gammas, xis, seq_lls = zip(*results)
        total_ll = sum(seq_lls)

        return list(gammas), list(xis), total_ll

    def maximization_step(self, observations, actions, gammas, xis, iteration=0):
        """M-step: Update model parameters based on expected sufficient statistics"""
        # Number of sequences
        n_sequences = len(observations)

        # Update initial state probabilities
        self.initial_prob = np.zeros(self.n_states)
        for i in range(n_sequences):
            self.initial_prob += gammas[i][0]
        self.initial_prob /= n_sequences

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

    def fit(self, observations, actions, max_iterations=100, tolerance=1e-4):
        """Run EM algorithm to fit the POMDP model parameters"""
        prev_ll = -np.inf
        sleep(1)
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

    def _ensure_psd(self, sigma, min_val=None):
        """Ensure covariance matrix is positive semi-definite"""
        if min_val is None:
            min_val = self.epsilon_prior

        # Symmetrize add regularization to diagonal
        sigma = 0.5 * (sigma + sigma.T)
        sigma = sigma + min_val * np.eye(sigma.shape[0])

        return sigma
    #TODO: refactor this function to be cleaner
    #TODO: move to a visualization module
    def compare_state_transitions(self, learned_transitions, baseline_transitions, baseline_observations, states_list,
                                  state_mapping=None):
        """
        Compares the transition probabilities for a user-selected state between
        a learned model and a baseline model.

        Args:
            learned_transitions (np.array): The learned transition probability matrix (n_states, n_actions, n_states).
            baseline_transitions (np.array): The baseline transition probability matrix (n_states, n_actions, n_states).
            states_list (list): A list of State objects, e.g., [State("healthy"), State("sick"), State("critical")].
        """
        selected_state_index = np.zeros(len(states_list), dtype=int)

        if state_mapping is None:
            print("\nSelect a state to compare transition probabilities:")
            for i, state in enumerate(states_list):
                print(f"{i}: {state.name}")

            s_count = 0
            for s in states_list:
                while True:
                    try:
                        choice = int(input(f"Enter the number for the state {s} (0-{len(states_list) - 1}): "))
                        if 0 <= choice < len(states_list):
                            selected_state_index[s_count] = choice
                            s_count += 1
                            break
                        else:
                            print("Invalid choice. Please enter a number within the valid range.")
                    except ValueError:
                        print("Invalid input. Please enter a number.")
        else:
            # Use provided mapping
            if isinstance(state_mapping, dict):
                for i, s in enumerate(states_list):
                    selected_state_index[i] = state_mapping.get(s.name, i)
            else:
                # Assume list/array in the correct order
                selected_state_index = np.array(state_mapping[:len(states_list)])

        def states_mapping(selected_state_index, s):
            """
            Maps the selected state name to its index in the states_list.
            """
            if s.name == "healthy":
                return selected_state_index[0]
            elif s.name == "sick":
                return selected_state_index[1]
            elif s.name == "critical":
                return selected_state_index[2]

            return None

        def mvn_pdf(x, weights, means, covariances):

            K = len(weights)
            pdf = 0.0

            for k in range(K):
                # Calculate the PDF for each Gaussian component
                component_pdf = weights[k] * multivariate_normal.pdf(x, mean=means[k], cov=covariances[k])
                pdf += component_pdf

            return pdf

        def mvn_kl(gmm_p, gmm_q, n_samples=1e5):
            X = gmm_p.sample(n_samples)[0]
            p_X = (mvn_pdf(X, gmm_p.weights_, gmm_p.means_, gmm_p.covariances_))
            q_X = (mvn_pdf(X, gmm_q.weights_, gmm_q.means_, gmm_q.covariances_))
            return np.mean(np.log(p_X / q_X))

        baseline_transitions_arr = np.zeros((3, 2, 3))
        for a in ["wait", "treat"]:
            for s in states_list:
                for s_prime in states_list:
                    index_s = states_mapping(selected_state_index, s)
                    index_s_prime = states_mapping(selected_state_index, s_prime)
                    a_index = 0 if a == "wait" else 1
                    baseline_transitions_arr[index_s, a_index, index_s_prime] = \
                        baseline_transitions[a, s.name, s_prime.name]

        # Compute normalized L1 distance (dividing by total number of elements to scale between 0 and 1)
        l1_distance = np.sum(np.abs(learned_transitions - baseline_transitions_arr)) / learned_transitions.size

        print(f"L1 distance for transitions: {l1_distance:.4f}")

        # Compute KL divergence between observation distributions
        kl_divergences = np.zeros((len(states_list)))
        obs_model_real = baseline_observations
        for s in range(len(states_list)):
            state_name = states_list[s].name

            # Get mapped state indices
            real_s = states_mapping(selected_state_index, states_list[s])

            means = []
            variances = []
            for dim, p in obs_model_real.params[state_name].items():
                a, b = p
                # Beta mean and variance
                means.append(a / (a + b))
                variances.append((a * b) / ((a + b) ** 2 * (a + b + 1)))

            real_GMM = GaussianMixture(n_components=1, means_init=means)
            real_GMM.means_ = np.array([means])
            real_GMM.weights_ = [1]  # Assuming equal weights for simplicity
            real_GMM.covariances_ = [np.diag(variances), np.diag(variances)]
            # Create learned GMM from the learned parameters
            learned_GMM = GaussianMixture(n_components=1, means_init=self.obs_means[real_s])
            learned_GMM.means_ = self.obs_means[real_s].reshape(1, -1)
            learned_GMM.weights_ = [1]  # Assuming equal weights for simplicity
            learned_GMM.covariances_ = [self.obs_covs[real_s], self.obs_covs[real_s]]

            # Compute KL divergence
            kl_divergences[s] = mvn_kl(real_GMM, learned_GMM)
            print(f"KL Divergences for {state_name}: {kl_divergences[s]:.4f}")

        result = {
            "l1_distance": l1_distance,
            "kl_divergences": np.average(kl_divergences)
        }

        return result


#TODO: move to a visualization module
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
