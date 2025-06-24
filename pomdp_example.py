import random

import numpy as np
import pomdp_py
from pomdp_py import vi_pruning
from sklearn.metrics import r2_score

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
    def __init__(self, params=None):
        super().__init__()
        if params is None:
            self.transitions = DEFAULT_TRANS_PARAMS
        else:
            self.transitions = params

        self.states = [State(s) for s in ["healthy", "sick", "critical"]]

    def probability(self, next_state, state, action):
        return self.transitions.get((action.name, state.name, next_state.name), 0.0)

    def sample(self, state, action):
        states = ['healthy', 'sick', 'critical']
        probs = [self.transitions.get((action.name, state.name, s), 0.0) for s in states]
        next_state_name = random.choices(states, weights=probs, k=1)[0]
        return State(next_state_name)

    def get_all_states(self):
        return self.states


# Observation model
class MedicalObservationModel(pomdp_py.ObservationModel):
    def __init__(self):
        self.observations = {
            # Healthy state observations
            ('healthy', 'o11'): 0.00,
            ('healthy', 'o10'): 0.1,
            ('healthy', 'o01'): 0.01,
            ('healthy', 'o00'): 0.89,

            # Sick state observations
            ('sick', 'o11'): 0.06,
            ('sick', 'o10'): 0.10,
            ('sick', 'o01'): 0.70,
            ('sick', 'o00'): 0.14,

            # Critical state observations
            ('critical', 'o11'): 0.90,
            ('critical', 'o10'): 0.03,
            ('critical', 'o01'): 0.05,
            ('critical', 'o00'): 0.02
        }
        self.all_observations = [Observation(o) for o in ['o11', 'o10', 'o01', 'o00']]

    def probability(self, observation, next_state, action):
        return self.observations.get((next_state.name, observation.name), 0.0)

    def sample(self, next_state, action):
        obs_list = ['o11', 'o10', 'o01', 'o00']
        probs = [self.observations.get((next_state.name, o), 0.0) for o in obs_list]
        obs_name = random.choices(obs_list, weights=probs, k=1)[0]
        return Observation(obs_name)

    def get_all_observations(self):
        return self.all_observations


# Reward model
class MedicalRewardModel(pomdp_py.RewardModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rewards = {
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

    def _reward(self, state, action, next_state, observation=None):
        return self.rewards.get((action.name, state.name, next_state.name), 0.0)

    def sample(self, state, action, next_state):
        return self._reward(state, action, next_state)


# Policy model
class MedicalPolicyModel(pomdp_py.RolloutPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.actions = [MedAction(a) for a in ["wait", "treat"]]

    def sample(self, state):
        return random.choice(self.actions)

    def get_all_actions(self, state=None, history=None):
        return self.actions

    def rollout(self, state, *args):
        """Treating this PolicyModel as a rollout policy"""
        return self.sample(state)


class FuzzyRewardModel(pomdp_py.RewardModel):
    def __init__(self, base_reward_model, fuzzy_model, weight=0.5):
        self.base = base_reward_model
        self.fuzzy = fuzzy_model
        self.alpha = weight  # fraction of fuzzy reward
        self.obs_model = MedicalObservationModel()

    def sample(self, state, action, next_state):
        # original reward
        base_r = self.base.sample(state, action, next_state)
        # map state→obs for fuzzy input
        obs_name = max([self.obs_model.sample(state, action).name
                        for _ in range(30)],
                       key=lambda x: [self.obs_model.sample(state, action).name for _ in range(30)].count(x))
        obs = Observation(obs_name)
        # fuzzy prediction
        _, fuzzy_r = predict_next_observation_reward(self.fuzzy, obs, action)

        # blended reward
        return (1 - self.alpha) * base_r + self.alpha * fuzzy_r


# Create the POMDP
def create_medical_pomdp(use_fuzzy=False, init_state=None):
    transition_model = MedicalTransitionModel()
    observation_model = MedicalObservationModel()
    base_reward = MedicalRewardModel()
    fuzzy_model =  None
    if use_fuzzy:
        # Use fuzzy model for reward prediction
        reward_model = FuzzyRewardModel(base_reward, fuzzy_model, weight=0.2)
    else:
        # Use standard reward model
        reward_model = MedicalRewardModel()

    policy_model = MedicalPolicyModel()
    init_belief = pomdp_py.Histogram({
        State("healthy"): 1 / 3,
        State("sick"): 1 / 3,
        State("critical"): 1 / 3
    })
    agent = pomdp_py.Agent(init_belief, policy_model,
                           transition_model,
                           observation_model,
                           reward_model,
                           name="MedicalAgent")
    if init_state is None:
        init_state = State(random.choice(["healthy", "sick", "critical"]))
    else:
        init_state = State(init_state)
    env = pomdp_py.Environment(init_state, transition_model, reward_model)
    return pomdp_py.POMDP(agent, env)


# Run the POMDP with a solver
def solve_medical_pomdp(horizon=8):
    pomdp = create_medical_pomdp()
    agent = pomdp.agent
    env = pomdp.env

    # Create a POUCT solver (Monte Carlo Tree Search-based)
    pouct = pomdp_py.POUCT(
        max_depth=horizon,
        discount_factor=0.95,
        num_sims=100,
        exploration_const=5,
        rollout_policy=agent.policy_model
    )

    # Run for several steps
    total_reward = 0
    for i in range(10):
        old_state = env.state
        # Agent selects action
        action = pouct.plan(agent)

        # Environment transitions and provides observation/reward
        #reward = env.reward_model.sample(env.state, action, None)
        reward = env.state_transition(action, execute=True)
        new_state = env.state
        observation = agent.observation_model.sample(env.state, action)
        #print("Reward:", reward)

        # Agent updates belief based on action and observation
        agent.update_history(action, observation)
        pouct.update(agent, action, observation)
        total_reward += reward

        previous_belief = agent.cur_belief

        if isinstance(agent.cur_belief, pomdp_py.Histogram):
            new_belief = pomdp_py.update_histogram_belief(agent.cur_belief,
                                                          action, observation,
                                                          agent.observation_model,
                                                          agent.transition_model)
            agent.set_belief(new_belief)

        # Print the current state, action, observation, and reward
        print(f"Step {i + 1}: Old State={old_state}, New State={new_state}")
        print(f"Previous Belief: {previous_belief}")
        print(f"Action={action}, Observation={observation}, Reward={reward}")
        print(f"Current Belief: {agent.cur_belief}")
        print("" + "-" * 50)

    print(f"Total reward: {total_reward}")
    return agent


def solve_medical_pomdp_with_comparison(pomdp, horizon=8, use_fuzzy=False, solver=None, random_action=False):
    """Run the POMDP with a solver and compare with a standard POMDP"""

    if pomdp is None:
        raise ValueError("POMDP cannot be None")

    if solver is None:
        solver = vi_pruning(pomdp.agent,
                            pomdp_solve_path='/home/loca/Downloads/pomdp-solve-5.5/src/pomdp-solve',
                            discount_factor=0.95,
                            options=["-horizon", "1000"],
                            remove_generated_files=False,
                            return_policy_graph=False
                            )

    agent = pomdp.agent
    env = pomdp.env

    pomdp_wf = create_medical_pomdp(use_fuzzy=False, init_state=env.state.name)

    total_reward = 0
    for i in range(10):
        old_state = env.state
        if random_action:
            action = random.choice(agent.policy_model.get_all_actions(env.state, agent.history))
        else:
            action = solver.plan(agent)

        env_reward = env.state_transition(action, execute=True)

        new_state = env.state
        observation = agent.observation_model.sample(env.state, action)
        reward = env_reward

        agent.update_history(action, observation)
        if use_fuzzy:
            # Take the reward without the fuzzy model
            reward = pomdp_wf.agent.reward_model.sample(old_state, action, new_state)

        total_reward += reward
        solver.update(agent, action, observation)

        if isinstance(agent.cur_belief, pomdp_py.Histogram):
            new_belief = pomdp_py.update_histogram_belief(agent.cur_belief,
                                                          action, observation,
                                                          agent.observation_model,
                                                          agent.transition_model)
            agent.set_belief(new_belief)
        print(f"Step {i + 1}: Old State={old_state}, New State={new_state}")
        print(f"Action={action}, Observation={observation}, Reward={reward}")
        print(f"Current Belief: {agent.cur_belief}")
        print("-" * 50)
    print(f"Total reward: {total_reward}")
    return total_reward, agent


def get_most_likely_observation(agent):
    """Extract the most likely observation from the belief state"""
    if not isinstance(agent.cur_belief, pomdp_py.Histogram):
        return "o00"  # Default

    # Get most likely state
    most_likely_state = max(agent.cur_belief.histogram.items(), key=lambda x: x[1])[0]

    # Map states to likely observations (simplistic mapping for illustration)
    state_to_obs = {
        "healthy": "o00",  # Healthy typically shows negative test and no symptoms
        "sick": "o10",  # Sick typically shows positive test but may not have symptoms
        "critical": "o11"  # Critical typically shows positive test and symptoms
    }

    return state_to_obs.get(most_likely_state.name, "o00")


def compare_models(horizon=8, trials=50):
    """Run both models and compare performance"""
    standard_rewards = []
    fuzzy_rewards = []
    random_rewards = []

    standard_pomdp = create_medical_pomdp(use_fuzzy=False)
    standard_solver = vi_pruning(standard_pomdp.agent,
                                 pomdp_solve_path='/home/loca/Downloads/pomdp-solve-5.5/src/pomdp-solve',
                                 discount_factor=0.9,
                                 options=["-horizon", "1000"],
                                 remove_generated_files=False,
                                 return_policy_graph=False
                                 )

    fuzzy_pomdp = create_medical_pomdp(use_fuzzy=True)
    fuzzy_solver = vi_pruning(fuzzy_pomdp.agent,
                              pomdp_solve_path='/home/loca/Downloads/pomdp-solve-5.5/src/pomdp-solve',
                              discount_factor=0.9,
                              options=["-horizon", "1000"],
                              remove_generated_files=False,
                              return_policy_graph=False
                              )

    for i in range(trials):
        #init state
        state = random.choice(["healthy", "sick", "critical"])
        print(f"\n--- Trial {i + 1}/{trials} ---")
        print("\nStandard POMDP:")
        # Reset the POMDP to its initial state
        standard_pomdp = create_medical_pomdp(use_fuzzy=False, init_state=state)
        std_reward, _ = solve_medical_pomdp_with_comparison(pomdp=standard_pomdp, horizon=horizon, use_fuzzy=False,
                                                            solver=standard_solver)
        standard_rewards.append(std_reward)

        print("\nFuzzy-enhanced POMDP:")
        # Reset the POMDP to its initial state
        fuzzy_pomdp = create_medical_pomdp(use_fuzzy=True, init_state=state)
        fuzzy_reward, _ = solve_medical_pomdp_with_comparison(pomdp=fuzzy_pomdp, horizon=horizon, use_fuzzy=True,
                                                              solver=fuzzy_solver)
        fuzzy_rewards.append(fuzzy_reward)

        random_pomdp = create_medical_pomdp(use_fuzzy=False, init_state=state)
        random_reward, _ = solve_medical_pomdp_with_comparison(pomdp=random_pomdp, horizon=horizon, use_fuzzy=False,
                                                               solver=standard_solver, random_action=True)
        random_rewards.append(random_reward)

    # Compare results
    print("\n=== COMPARISON RESULTS ===")
    print(f"Standard POMDP average reward: {sum(standard_rewards) / len(standard_rewards):.2f}")
    print(f"Fuzzy-enhanced POMDP average reward: {sum(fuzzy_rewards) / len(fuzzy_rewards):.2f}")
    print(f"Random action reward: {sum(random_rewards) / len(random_rewards):.2f}")
    print(
        f"Performance difference: {((sum(fuzzy_rewards) / len(fuzzy_rewards)) - (sum(standard_rewards) / len(standard_rewards))):.2f}")


def predict_next_observation_reward(fuzzy_model, current_observation, action):
    # Convert observation to test result and symptoms values
    test_val, symptom_val = fuzzy_model.obs_mapping[current_observation.name]
    action_val = fuzzy_model.action_mapping[action.name]

    # Pass values to fuzzy system
    fuzzy_model.prediction.input['action_input'] = action_val
    fuzzy_model.prediction.input['test_result'] = test_val
    fuzzy_model.prediction.input['symptoms'] = symptom_val

    # Compute result
    rsult = fuzzy_model.prediction.compute()

    if len(fuzzy_model.prediction.output) == 0:
        # Get outputs
        next_test_val = test_val
        next_symptom_val = symptom_val
        reward_val = 0.0
        fuzzy_model.default_count += 1
    else:
        # Get outputs
        next_test_val = fuzzy_model.prediction.output['next_test']
        next_symptom_val = fuzzy_model.prediction.output['next_symptoms']
        normalized_reward = fuzzy_model.prediction.output['reward']

        # Denormalize reward
        reward_val = normalized_reward * (fuzzy_model.reward_max - fuzzy_model.reward_min) + fuzzy_model.reward_min

    # Convert continuous values to binary
    next_test_bool = next_test_val >= 0.5
    next_symptom_bool = next_symptom_val >= 0.5

    # Convert back to observation
    next_obs_name = fuzzy_model.inverse_mapping[(next_test_bool, next_symptom_bool)]
    next_observation = Observation(next_obs_name)

    return next_observation, reward_val


"""
def evaluate_fuzzy_reward_prediction(trials=100, horizon=20):

    base_reward_model = MedicalRewardModel()
    fuzzy_model = MedicalFuzzyModel()
    transition_model = MedicalTransitionModel()
    observation_model = MedicalObservationModel()

    true_rewards = []
    fuzzy_predicted_rewards = []
    true_next_obs = []
    pred_next_obs = []
    all_actions = [MedAction(a) for a in ["wait", "treat"]]
    all_states = [State(s) for s in ["healthy", "sick", "critical"]]

    for _ in range(trials):
        current_state = random.choice(all_states)
        for _ in range(horizon):
            action = random.choice(all_actions)
            # Sample next state and true next observation
            next_state = transition_model.sample(current_state, action)
            actual_obs = observation_model.sample(next_state, action)
            # True reward
            r_true = base_reward_model.sample(current_state, action, next_state)
            # Fuzzy prediction
            pred_obs, r_fuzzy = predict_next_observation_reward(fuzzy_model,
                                                                observation_model.sample(current_state, action),
                                                                action)
            # Record metrics
            true_rewards.append(r_true)
            fuzzy_predicted_rewards.append(r_fuzzy)
            true_next_obs.append(actual_obs.name)
            pred_next_obs.append(pred_obs.name)
            current_state = next_state

    # Convert to numpy arrays
    y_true = np.array(true_rewards)
    y_pred = np.array(fuzzy_predicted_rewards)

    # Reward metrics
    if np.var(y_true) == 0 or np.var(y_pred) == 0:
        r2 = 0.0 if np.var(y_true) == 0 else r2_score(y_true, y_pred)
        r = 0.0
        print("Warning: Constant data encountered in rewards, metrics might be misleading.")
    else:
        r2 = r2_score(y_true, y_pred)
        r = np.corrcoef(y_true, y_pred)[0, 1]

    # Observation prediction accuracy
    obs_accuracy = np.mean([t == p for t, p in zip(true_next_obs, pred_next_obs)])

    # Print results
    print("\n--- Fuzzy Reward & Observation Prediction Performance ---")
    print(f"Total steps: {trials * horizon}")
    print(f"Reward R²: {r2:.4f}, Pearson’s r: {r:.4f}")
    print(f"Next-observation prediction accuracy: {obs_accuracy:.4f}")

    return r2, r, obs_accuracy
"""

# Example usage (assuming this code is added to or run alongside main_fever.py)
if __name__ == "__main__":
    print("Running Medical POMDP Example...")
    # compare_models(horizon=10, trials=3) # Keep or remove existing main execution
    # evaluate_fuzzy_true_reward(trials=30, horizon=10) # Keep or remove existing main execution
    #evaluate_fuzzy_reward_prediction(trials=500, horizon=10)
    #compare_models(horizon=10, trials=50)
#    solve_medical_pomdp(horizon=120)

#print("Test the interfaces with Cassandra")
#pomdp = create_medical_pomdp()
#agent = pomdp.agent
#env = pomdp.env
#
#pomdp_solve_path = '/home/loca/Downloads/pomdp-solve-5.5/src/pomdp-solve'
#policy = vi_pruning(agent, pomdp_solve_path, discount_factor=0.9,
#                    options=["-horizon", "100",
#                             "-epsilon", "0.01", "-stop_delta", "0.01", "-proj_purge", "domonly",
#                             "-prune_epsilon", "1e-3"],
#                    remove_generated_files=False,
#                    return_policy_graph=False
#                    )
