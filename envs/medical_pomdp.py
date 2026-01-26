import random
import pomdp_py

DEFAULT_TRANS_PARAMS = {
    # Wait action transitions
    ('wait', 'healthy', 'healthy'): 0.85,
    ('wait', 'healthy', 'sick'): 0.14,
    ('wait', 'healthy', 'critical'): 0.01,
    ('wait', 'sick', 'healthy'): 0.3,
    ('wait', 'sick', 'sick'): 0.6,
    ('wait', 'sick', 'critical'): 0.1,
    ('wait', 'critical', 'healthy'): 0.01,
    ('wait', 'critical', 'sick'): 0.05,
    ('wait', 'critical', 'critical'): 0.94,

    # Treat action transitions
    ('treat', 'healthy', 'healthy'): 0.8,
    ('treat', 'healthy', 'sick'): 0.15,
    ('treat', 'healthy', 'critical'): 0.05,
    ('treat', 'sick', 'healthy'): 0.65,
    ('treat', 'sick', 'sick'): 0.35,
    ('treat', 'sick', 'critical'): 0.0,
    ('treat', 'critical', 'healthy'): 0.1,
    ('treat', 'critical', 'sick'): 0.65,
    ('treat', 'critical', 'critical'): 0.25,
}
DEFAULT_OBS_PARAMS = {
    ('healthy', 'o11'): 0.00,
    ('healthy', 'o10'): 0.1,
    ('healthy', 'o01'): 0.01,
    ('healthy', 'o00'): 0.89,

    ('sick', 'o11'): 0.06,
    ('sick', 'o10'): 0.10,
    ('sick', 'o01'): 0.70,
    ('sick', 'o00'): 0.14,

    ('critical', 'o11'): 0.90,
    ('critical', 'o10'): 0.03,
    ('critical', 'o01'): 0.05,
    ('critical', 'o00'): 0.02
}
DEFAULT_STATES = ["healthy", "sick", "critical"]
DEFAULT_ACTIONS = ["wait", "treat"]
DEFAULT_REWARD = {
    # wait rewards
    ('wait', 'healthy', 'healthy'): +20,  # Increased reward for staying healthy
    ('wait', 'healthy', 'sick'): -5,  # Increased penalty for deterioration
    ('wait', 'healthy', 'critical'): -15,  # Doubled penalty for severe deterioration
    ('wait', 'sick', 'healthy'): +10,  # Increased reward for natural recovery
    ('wait', 'sick', 'sick'): -7,  # Slightly higher penalty for no improvement
    ('wait', 'sick', 'critical'): -20,  # Increased penalty for worsening
    ('wait', 'critical', 'healthy'): +15,  # Increased reward for miraculous recovery
    ('wait', 'critical', 'sick'): +5,  # Now positive reward for partial improvement
    ('wait', 'critical', 'critical'): -30,  # Higher penalty for staying critical

    # treat rewards
    ('treat', 'healthy', 'healthy'): -15,  # Reduced penalty for unnecessary treatment
    ('treat', 'healthy', 'sick'): -20,  # Increased penalty for negative outcome
    ('treat', 'healthy', 'critical'): -25,  # Increased penalty for severe negative outcome
    ('treat', 'sick', 'healthy'): +25,  # Increased reward for successful treatment
    ('treat', 'sick', 'sick'): -2,  # Small penalty for ineffective treatment
    ('treat', 'sick', 'critical'): -18,  # Increased penalty for harmful treatment
    ('treat', 'critical', 'healthy'): +40,  # Significantly increased reward for saving critical patient
    ('treat', 'critical', 'sick'): +20,  # Doubled reward for improving critical condition
    ('treat', 'critical', 'critical'): -15  # Reduced penalty as treatment attempt was appropriate
}


# Define state, action, observation classes
class State(pomdp_py.State):
    def __init__(self, name):
        self.name = name

    def __hash__(self): return hash(self.name)

    def __eq__(self, other): return isinstance(other, State) and self.name == other.name

    def __str__(self): return self.name

    def __repr__(self): return self.name


class MedAction(pomdp_py.Action):
    def __init__(self, name):
        self.name = name

    def __hash__(self): return hash(self.name)

    def __eq__(self, other): return isinstance(other, MedAction) and self.name == other.name

    def __str__(self): return self.name


class Observation(pomdp_py.Observation):
    def __init__(self, name):
        self.name = name

    def __hash__(self): return hash(self.name)

    def __eq__(self, other): return isinstance(other, Observation) and self.name == other.name

    def __str__(self):
        return self.name


# Transition model
class MedicalTransitionModel(pomdp_py.TransitionModel):
    def __init__(self, states, transition=None):
        super().__init__()
        if transition is None:
            self.transitions = DEFAULT_TRANS_PARAMS
        else:
            self.transitions = transition

        self.states = [State(s) for s in states]

    def probability(self, next_state, state, action):
        return self.transitions.get((action.name, state.name, next_state.name), 0.0)

    def sample(self, state, action):
        state_names = [s.name for s in self.states]
        probs = [self.transitions.get((action.name, state.name, s), 0.0) for s in state_names]
        next_state_name = random.choices(state_names, weights=probs, k=1)[0]
        return State(next_state_name)

    def get_all_states(self):
        return self.states

# Observation model
class MedicalObservationModel(pomdp_py.ObservationModel):
    def __init__(self, observations=None):
        """
        General observation model.
        :param observations: Dizionario delle probabilità di osservazione.
                              Esempio: {("state", "observation"): prob}
        """
        if observations is None:
            observations = DEFAULT_OBS_PARAMS
        self.observations = observations
        self.all_observations = list({Observation(o) for _, o in observations.keys()})

    def probability(self, observation, next_state, action):
        return self.observations.get((next_state.name, observation.name), 0.0)

    def sample(self, next_state, action):
        obs_list = list({o for _, o in self.observations.keys() if _ == next_state.name})
        probs = [self.observations.get((next_state.name, o), 0.0) for o in obs_list]
        obs_name = random.choices(obs_list, weights=probs, k=1)[0]
        return Observation(obs_name)

    def get_all_observations(self):
        return self.all_observations


# Reward model
class MedicalRewardModel(pomdp_py.RewardModel):
    def __init__(self, reward=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if reward is None:
            reward = DEFAULT_REWARD
        self.rewards = reward

    def _reward(self, state, action, next_state, observation=None):
        return self.rewards.get((action.name, state.name, next_state.name), 0.0)

    def sample(self, state, action, next_state):
        return self._reward(state, action, next_state)


# Policy model
class MedicalPolicyModel(pomdp_py.RolloutPolicy):
    def __init__(self, actions=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        actions = actions if actions is not None else DEFAULT_ACTIONS
        self.actions = actions

    def sample(self, state, med_action_return=False):
        return random.choice(self.actions) if not med_action_return else MedAction(random.choice(self.actions))

    def get_all_actions(self, state=None, history=None):
        return self.actions

    def rollout(self, state, *args):
        """Treating this PolicyModel as a rollout policy"""
        return self.sample(state)


class FuzzyRewardModel(pomdp_py.RewardModel):
    def __init__(self, base_reward_model, fuzzy_model, weight=0.5):
        self.base = base_reward_model
        self.fuzzy = fuzzy_model
        self.alpha = weight
        self.obs_model = MedicalObservationModel()

    def sample(self, state, action, next_state):
        base_r = self.base.sample(state, action, next_state)
        obs_name = max([self.obs_model.sample(state, action).name
                        for _ in range(30)],
                       key=lambda x: [self.obs_model.sample(state, action).name for _ in range(30)].count(x))
        obs = Observation(obs_name)

        _, fuzzy_r = self.predict_next_observation_reward(self.fuzzy, obs, action)

        return (1 - self.alpha) * base_r + self.alpha * fuzzy_r

    def predict_next_observation_reward(self, current_observation, action):
        test_val, symptom_val = self.obs_mapping[current_observation.name]
        action_val = self.action_mapping[action.name]

        self.prediction.input['action_input'] = action_val
        self.prediction.input['test_result'] = test_val
        self.prediction.input['symptoms'] = symptom_val

        result = self.prediction.compute()

        if len(self.prediction.output) == 0:
            next_test_val = test_val
            next_symptom_val = symptom_val
            reward_val = 0.0
            self.default_count += 1
        else:
            next_test_val = self.prediction.output['next_test']
            next_symptom_val = self.prediction.output['next_symptoms']
            normalized_reward = self.prediction.output['reward']

            reward_val = normalized_reward * (self.reward_max - self.reward_min) + self.reward_min

        next_test_bool = next_test_val >= 0.5
        next_symptom_bool = next_symptom_val >= 0.5

        next_obs_name = self.inverse_mapping[(next_test_bool, next_symptom_bool)]
        next_observation = Observation(next_obs_name)

        return next_observation, reward_val


if __name__ == "__main__":
    print("Running Medical POMDP Example...")
