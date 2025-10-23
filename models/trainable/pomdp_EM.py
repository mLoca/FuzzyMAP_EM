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

    def __init__(self, n_states: int, n_actions: int, obs_dim: int, verbose: bool = False, seed: int = 42, parallel: bool = True):
        """Initialize the POMDP model with EM capabilities"""
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

    def observation_likelihood(self, obs, state):
        """Compute P(o|s) using multivariate Gaussian"""
        return multivariate_normal.pdf(
            obs,
            mean=self.obs_means[state],
            cov=self.obs_covs[state]
        )

    def forward_pass(self, observations, actions):
        """Forward algorithm computing α_t(s)"""
        T = len(observations)
        alpha = np.zeros((T, self.n_states))
        c = np.zeros(T)

        # Initial forward probability
        for s in range(self.n_states):
            alpha[0, s] = self.initial_prob[s] * self.observation_likelihood(observations[0], s)

        # Normalize to prevent numerical underflow
        c[0] = np.sum(alpha[0])
        if c[0] > 0:
            alpha[0] /= c[0]

        # Forward recursion
        for t in range(1, T):
            for s_prime in range(self.n_states):
                for s in range(self.n_states):
                    alpha[t, s_prime] += alpha[t - 1, s] * self.transitions[s, actions[t - 1], s_prime]
                alpha[t, s_prime] *= self.observation_likelihood(observations[t], s_prime)

            c[t] = np.sum(alpha[t])
            alpha[t] /= c[t] + 1e-20 # Avoid division by zero

        log_likelihood = np.sum(np.log(c))

        return alpha, log_likelihood

    def backward_pass(self, observations, actions):
        """Backward algorithm computing β_t(s)"""
        T = len(observations)
        beta = np.zeros((T, self.n_states))

        # Initialize with beta_(T-1) = 1
        beta[T - 1] = np.ones(self.n_states)

        # Backward recursion
        for t in range(T - 2, -1, -1):
            for s in range(self.n_states):
                for s_prime in range(self.n_states):
                    beta[t, s] += self.transitions[s, actions[t], s_prime] * \
                                  self.observation_likelihood(observations[t + 1], s_prime) * \
                                  beta[t + 1, s_prime]

            beta[t] /= np.sum(beta[t]) + 1e-10 # Normalize to prevent underflow & avoid division by zero

        return beta

    def compute_posterior(self, alpha, beta):
        """Compute γ_t(s) = P(s_t=s|o_1:T, a_1:T)"""
        gamma = alpha * beta
        gamma /= np.sum(gamma, axis=1, keepdims=True) + 1e-10
        return gamma

    def compute_transition_counts(self, alpha, beta, observations, actions):
        """Compute ξ_t(s,s') = P(s_t=s, s_{t+1}=s'|o_1:T, a_1:T)"""
        T = len(observations)
        xi = np.zeros((T - 1, self.n_states, self.n_states))

        for t in range(T - 1):
            for s in range(self.n_states):
                for s_prime in range(self.n_states):
                    xi[t, s, s_prime] = alpha[t, s] * \
                                        self.transitions[s, actions[t], s_prime] * \
                                        self.observation_likelihood(observations[t + 1], s_prime) * \
                                        beta[t + 1, s_prime]

            xi[t] /= np.sum(xi[t]) + 1e-10

        return xi

    def expectation_step(self, observations, actions):
        """E-step: Compute expected sufficient statistics"""
        gammas = []
        xis = []
        total_ll = 0.0

        for obs_seq, act_seq in zip(observations, actions):
            alpha, seq_ll = self.forward_pass(obs_seq, act_seq)
            total_ll += seq_ll

            beta = self.backward_pass(obs_seq, act_seq)
            gamma = self.compute_posterior(alpha, beta)
            xi = self.compute_transition_counts(alpha, beta, obs_seq, act_seq)

            gammas.append(gamma)
            xis.append(xi)

        return gammas, xis, total_ll

    #e-step with parallelization
    def expectation_step_parallel(self, observations, actions):
        """Parallelized E-step using multi-processing"""
        import multiprocessing as mp

        with mp.Pool(processes=mp.cpu_count()) as pool:
            # Process each observation sequence in parallel
            results = pool.starmap(
                self._process_single_sequence,
                [(obs_seq, act_seq) for obs_seq, act_seq in zip(observations, actions)]
            )

        # Unpack results
        gammas, xis, seq_lls = zip(*results)
        total_ll = sum(seq_lls)

        return list(gammas), list(xis), total_ll

    def _process_single_sequence(self, obs_seq, act_seq):
        """Process a single observation sequence for parallelization"""
        alpha, seq_ll = self.forward_pass(obs_seq, act_seq)
        beta = self.backward_pass(obs_seq, act_seq)
        gamma = self.compute_posterior(alpha, beta)
        xi = self.compute_transition_counts(alpha, beta, obs_seq, act_seq)
        return gamma, xi, seq_ll

    def maximization_step(self, observations, actions, gammas, xis):
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
                total_weight = obs_counts[s] + 1e-10
                self.obs_means[s] = new_means[s] / total_weight
                # Update covariance
                E_ooT_s = data_O_sum_gamma_obs_sq[s] / total_weight
                mu_s_outer_mu_s = np.outer(self.obs_means[s, :], self.obs_means[s, :])
                self.obs_covs[s, :, :] = E_ooT_s - mu_s_outer_mu_s
                self.obs_covs[s, :, :] += 1e-8 * np.eye(self.obs_dim)  # Regularization

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

                # M-step
                self.maximization_step(observations, actions, gammas, xis)

                # Check convergence
                improvement = log_likelihood - prev_ll
                prev_ll = log_likelihood

                if self.verbose:
                    print(
                        f"Iteration {iteration + 1}, Log-Likelihood: {log_likelihood:.4f}, Improvement: {improvement:.4f}")

                if np.abs(improvement) < tolerance:
                    print(f"Converged after {iteration + 1} iterations")
                    break
        except ValueError as e:
            print(f"An error occurred during EM fitting: {e}")
            return 0
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return 0

        return log_likelihood

    #TODO: refactor this function to be cleaner
    #TODO: move to a visualization module
    def compare_state_transitions(self, learned_transitions, baseline_transitions, baseline_observations, states_list, state_mapping=None):
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

            a_t, b_t, a_s, b_s = obs_model_real.params[state_name]
            # Beta mean and variance
            mean_t = a_t / (a_t + b_t)
            mean_s = a_s / (a_s + b_s)
            var_t = (a_t * b_t) / ((a_t + b_t) ** 2 * (a_t + b_t + 1))
            var_s = (a_s * b_s) / ((a_s + b_s) ** 2 * (a_s + b_s + 1))

            real_GMM = GaussianMixture(n_components=1, means_init=[mean_t, mean_s])
            real_GMM.means_ = np.array([[mean_t, mean_s]])
            real_GMM.weights_ = [1]  # Assuming equal weights for simplicity
            real_GMM.covariances_ = [np.diag([var_t, var_s]), np.diag([var_t, var_s])]
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



#TODO: move to a training module
def train_pomdp_model(observations, actions, n_states, n_actions, obs_dim, max_iterations, tolerance):
    """Train a POMDP model using the provided data"""

    print(f"Training  POMDP model...")

    model = PomdpEM(
        n_states=n_states,
        n_actions=n_actions,
        obs_dim=obs_dim,
        verbose=True)

    log_likelihood = model.fit(
        observations,
        actions,
        max_iterations=max_iterations,
        tolerance=tolerance
    )

    print(f" POMDP training completed. Final log-likelihood: {log_likelihood:.4f}")
    return model, log_likelihood

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

#TODO: move to a training module
def run_pomdp_reconstruction():
    """Main experiment function for POMDP reconstruction"""
    # Parameters
    n_trajectories = 15
    trajectory_length = 5
    n_states = 3  # healthy, sick, critical
    n_actions = 2  # wait, treat
    obs_dim = 2  # test_result and symptoms (continuous)
    seed = 42  # For reproducibility

    # Generate training data
    original_pomdp, observations, actions, true_states, rewards = generate_pomdp_data(
        n_trajectories, trajectory_length, seed=seed
    )

    # Train standard POMDP model
    standard_pomdp, standard_ll = train_pomdp_model(
        observations, actions, n_states=n_states, n_actions=n_actions, obs_dim=obs_dim,
        max_iterations=200, tolerance=1e-8,
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
        original_pomdp.agent.observation_model,
        STATES
    )

    return standard_pomdp


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
    #evaluate_fuzzy_reward_prediction()
    #sensitivity_results = rho_sensitivity_analysis()
