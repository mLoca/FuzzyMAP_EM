from POMDPPlanners.planners.mcts_planners.pft_dpw import PFT_DPW
from POMDPPlanners.utils.action_samplers import DiscreteActionSampler
from POMDPPlanners.core.environment import (
    Environment,
    DiscreteActionsEnvironment,
    SpaceInfo,
    SpaceType,
)
from POMDPPlanners.core.belief import get_initial_belief
from POMDPPlanners.utils.belief_factory import create_environment_belief
from POMDPPlanners.simulations.simulation_apis.local_simulations_api import LocalSimulationsAPI
from POMDPPlanners.core.simulation import EnvironmentRunParams

import numpy as np
import scipy.stats 
from scipy.stats import multivariate_normal

eps_values_for_actions = np.array([[0., 0.], [.7, 0.], [0., .3], [.7, .3]])

class DiscreteStateDistribution:
    def __init__(self, probabilities):
        self.probabilities = probabilities
        self.n_states = len(probabilities)

    def sample(self, n_samples=1):
        # POMDPPlanners calls this method to generate the initial particle belief
        return np.random.choice(self.n_states, size=n_samples, p=self.probabilities)


class LearnedHIVEnvironment(Environment):
    def __init__(self, name, learned_params, hiv_reward_function):
        self.space_info = SpaceInfo(SpaceType.DISCRETE, SpaceType.CONTINUOUS)
        super().__init__(0.95, name, self.space_info)
        self.T = learned_params["T"]          # Shape: (n_states, n_actions, n_states)
        self.mu = learned_params["mu"]        # Shape: (n_states, obs_dim)
        self.Sigma = learned_params["Sigma"]  # Shape: (n_states, obs_dim, obs_dim)
        
        # The reward function wasn't learned, so we pass the benchmark's true reward
        self.hiv_reward_function = hiv_reward_function 
        
        self.n_states = self.T.shape[0]
        self.n_actions = self.T.shape[1]
        self.obs_dim = self.mu.shape[1]

    def sample_next_state(self, state, action):
        # Sample next state using the learned transition matrix T(s, a, s')
        transition_probs = self.T[state, action, :]
        return np.random.choice(self.n_states, p=transition_probs)

    def sample_observation(self, next_state, action):
        # Sample observation from the learned Multivariate Normal distribution O(o|s)
        state = next_state
        mean = self.mu[state]
        cov = self.Sigma[state]
        return np.random.multivariate_normal(mean, cov)

    def reward(self, state, action, next_state= 0):
        # Delegate to the HIV benchmark's reward function
        #return self.hiv_reward_function(state, action, next_state)
        expected_obs = self.mu[state]

        eps1, eps2 = eps_values_for_actions[action]

        # expected_obs is log10 AND standardized.e.
        mean_V = 3.7256933
        std_V = 1.16411527
        mean_E = 1.73064582
        std_E = 0.20234819

        log_V = expected_obs[2] * std_V + mean_V
        log_E = expected_obs[3] * std_E + mean_E
        
        V = 10 ** log_V
        E = 10 ** log_E
        
        reward = (-0.1 * V) - 2e4 * eps1 ** 2 - 2e3 * eps2 ** 2 + (1e3 * E)
        return  reward

    def is_terminal(self, state):
        # Define termination conditions for the HIV benchmark
        # (e.g., patient reaches a specific critical state or maximum horizon)
        return False 

    def initial_state_dist(self):
        # Create a uniform distribution over all states
        probs = np.ones(self.n_states) / self.n_states
        return DiscreteStateDistribution(probs)
    
    def hash_action(self, action):
        # Actions are discrete, so we can just return them as integers or tuples.
        # This is used by the planner to index action nodes in the search tree.
        return int(action)

    def is_equal_observation(self, obs1, obs2):
        # The planner needs to know if two observations match to group them in the tree.
        # If you are using continuous observations (PFT-DPW), we use a tolerance check.
        # If you discretized them for POMCP, this handles exact tuple matching as well.
        return np.allclose(obs1, obs2)

    def transition_log_probability(self, state, action, next_states):
        # Returns the log probability of transitioning to next_state: log(T[s, a, s'])
        result = np.full(len(next_states), -np.inf)
        for i, ns in enumerate(next_states):
            result[i] = np.log(self.T[state, action, next_states[i]])
        # We use max(prob, 1e-12) to prevent taking log(0) which results in -inf
        return result

    def observation_log_probability(self, action, next_state, observations):
        # Returns log(O[o | s']). In Fuzzy-MAP EM, the observation depends on the state.
        # We evaluate the log-pdf of the Multivariate Normal distribution.
        mean = self.mu[next_state]
        cov = self.Sigma[next_state]
        cov_safe = cov + np.eye(self.obs_dim) * 1e-6
        return [multivariate_normal.logpdf(observations, mean, cov_safe)]
        

    def initial_observation_dist(self):
        # Used by exact belief trackers to initialize the observation probability space.
        # Since particle filters (which you'll use for continuous spaces) rely on 
        # generative sampling instead of exact distributions, we can safely return None 
        # or a uniform placeholder without breaking the simulation.
        return None

    def get_actions(self):
        # Returns a list of all valid action indices [0, 1, ..., n_actions - 1]
        return list(range(self.n_actions))

