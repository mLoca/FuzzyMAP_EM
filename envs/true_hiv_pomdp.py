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
        # Extract the continuous 6D state
        arrival_state = next_state
        
        T1 = arrival_state[0]
        T2 = arrival_state[1]
        V  = arrival_state[4]
        E  = arrival_state[5]

        return np.array([T1, T2, V, E], dtype=float)

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