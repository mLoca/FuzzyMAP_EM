import numpy as np
from POMDPPlanners.core.environment import (Environment, SpaceType, SpaceInfo)

# Import your actual simulator class
from hiv_simulator import HIVSimulator

class TrueHIVStateDistribution:
    """Always starts the True Patient at the baseline unhealthy state."""
    def __init__(self, initial_state):
        self.initial_state = np.array(initial_state, dtype=float)

    def sample(self, n_samples=1):
        # Returns shape (n_samples, 6)
        return np.tile(self.initial_state, (n_samples, 1))

class TrueHIVEnvironment(Environment):
    def __init__(self, initial_biological_state):
        self.space_info = SpaceInfo(SpaceType.DISCRETE, SpaceType.CONTINUOUS)
        super().__init__(0.95, "true_hiv_env", self.space_info)
        # True biological state is 6D: [T1, T1*, T2, T2*, V, E]
        self.initial_biological_state = np.array(initial_biological_state, dtype=float)
        self.n_actions = 4 
        
        # Instantiate ONE internal simulator for this environment
        # (This is thread-safe for Joblib parallelization)
        self.sim = HIVSimulator(logspace=True)

    def get_actions(self):
        return list(range(self.n_actions))

    def hash_action(self, action):
        return int(action)

    def is_equal_observation(self, obs1, obs2):
        return np.allclose(obs1, obs2)

    def initial_state_dist(self):
        return TrueHIVStateDistribution(self.initial_biological_state)

    # ---------------------------------------------------------
    # THE TIE TO hiv_simulator.py (STATELESS OVERRIDE)
    # ---------------------------------------------------------
    def sample_next_state(self, state, action):
        """
        Forces the stateful HIVSimulator to act statelessly.
        """

        current_state = np.atleast_1d(state).astype(float)     
        if state.size == 1:
            current_state = np.copy(self.initial_biological_state)

        self.sim.state = np.copy(current_state)
        self.sim.t = 0.0 
        self.sim.step(action)
        next_state = np.copy(self.sim.state)
        
        return next_state

    # ---------------------------------------------------------
    # OBSERVATION & REWARD MAPPING
    # ---------------------------------------------------------
    def sample_observation(self, next_state, action):
        # We must return the standardized log observation to match the POMDP's training data.
        # The AI expects observations in standard normal form, not raw cell counts.
        mask = [True, True, False, False, True, True]
        empirical_mean = np.array([5.41947795, 1.59751311, 3.03701108, 1.45943164, 3.7256933, 1.73064582])
        empirical_std = np.array([0.1913442, 0.86731865, 1.19517352, 0.48159324, 1.16411527, 0.20234819])
        
        # Safely compute log10 of the physical state
        log_state = np.log10(np.clip(next_state, 1e-10, None))
        
        # Standardize using the same scalars as HIVSimulator
        standardized_obs = (log_state[mask] - empirical_mean[mask]) / empirical_std[mask]
        
        # Emulate clinical noise (matches training set args.noise = 0.1)
        obs = standardized_obs + np.random.normal(0, 0.1, size=standardized_obs.shape)
        
        return tuple(obs)

    def reward(self, state, action, next_state= 0):
        self.sim.logspace = False
        reward =  self.sim.calc_reward(action=action, state= next_state)
        self.sim.logspace = True
        return reward

    def is_terminal(self, state):
        return False

    # ---------------------------------------------------------
    # REQUIRED DUMMY METHODS FOR CONTINUOUS/GENERATIVE SPACES
    # ---------------------------------------------------------
    def transition_log_probability(self, state, action, next_states):
        return 0.0

    def observation_log_probability(self, next_state, action, observations):
        return 0.0

    def initial_observation_dist(self):
        return None