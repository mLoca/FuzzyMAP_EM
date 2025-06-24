import random

import pomdp_py
from pomdp_py import ObservationModel, Agent, Environment, POMDP, vi_pruning
import numpy as np
from scipy.stats import beta
from sklearn.metrics import r2_score
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from pomdp_example import State, MedAction, MedicalTransitionModel, MedicalRewardModel, Observation, \
    MedicalPolicyModel

STATES = [State(s) for s in ["healthy", "sick", "critical"]]
ACTIONS = [MedAction(a) for a in ["wait", "treat"]]
DEFAULT_OBS_PARAMS = {
    # Sano: Test per lo più negativo, sintomi per lo più assenti
    "healthy": (1.8, 7.5, 2, 10),
    # Malato: Test misto ma per lo più negativo, sintomi per lo più presenti
    "sick": (2.8, 5, 8, 2),
    # Critico: Test per lo più positivo, sintomi fortemente presenti
    "critical": (7.3, 2, 8.9, 2.5)
}
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


class ContinuousObservationModel(ObservationModel):
    def __init__(self, states, actions, obs_params=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.states = states
        self.actions = actions
        # define Beta distribution parameters per state
        # format: { state_name: (alpha_test, beta_test, alpha_symp, beta_symp) }
        if obs_params is None:
            self.params = DEFAULT_OBS_PARAMS
        else:
            self.params = obs_params

    def sample(self, next_state, action, **kwargs):
        a_t, b_t, a_s, b_s = self.params[next_state.name]
        test_obs = beta.rvs(a_t, b_t)
        symp_obs = beta.rvs(a_s, b_s)
        return np.array([test_obs, symp_obs])

    def probability(self, observation, next_state, action):
        a_t, b_t, a_s, b_s = self.params[next_state.name]
        test_obs, symp_obs = observation
        p_t = beta.pdf(test_obs, a_t, b_t)
        p_s = beta.pdf(symp_obs, a_s, b_s)
        return p_t * p_s

    def plot_observation_distribution(self):
        # Plot the observation distribution for each state
        # Create a grid of subplots (2 rows: tests and symptoms, 3 columns: states)
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        plt.suptitle("Observation Distribution for Each State", fontsize=16)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        # Loop through each state and plot the distributions
        colors = ['green', 'blue', 'red']
        for col, state in enumerate(self.states):
            state_name = state.name
            a_t, b_t, a_s, b_s = self.params[state_name]

            # Data for plotting
            x_test = np.linspace(0, 1, 100)
            x_symp = np.linspace(0, 1, 100)
            y_test = beta.pdf(x_test, a_t, b_t)
            y_symp = beta.pdf(x_symp, a_s, b_s)

            # Plot test results distribution (first row)
            axs[0, col].plot(x_test, y_test, color=colors[col])
            axs[0, col].set_title(f"{state_name}")
            axs[0, col].set_xlabel("Test Result")
            axs[0, col].set_ylabel("Density")

            # Plot symptoms distribution (second row)
            axs[1, col].plot(x_symp, y_symp, color=colors[col])
            axs[1, col].set_xlabel("Symptoms")
            axs[1, col].set_ylabel("Density")

        plt.tight_layout()
        plt.show()

        a_t, b_t, a_s, b_s = self.params["sick"]
        x_test = np.linspace(0, 1, 100)
        y_test = beta.pdf(x_test, a_t, b_t)
        plt.plot(x_test, y_test, label='Test Result', color='blue')
        plt.savefig("sick_test_distribution.png")


def generate_pomdp_data(n_trajectories, trajectory_length, seed=None):
    """Generate  data from POMDP model"""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    print("Generating training data...")
    observations = []
    actions = []
    true_states = []
    rewards = []

    for traj in range(n_trajectories):
        # Reset environment for new trajectory
        pomdp = create_continuous_medical_pomdp()
        pomdp.agent.set_belief(pomdp_py.Histogram({
            State("healthy"): 1 / 3, State("sick"): 1 / 3, State("critical"): 1 / 3
        }))

        traj_observations, traj_actions = [], []
        traj_states, traj_rewards = [], []

        for _ in range(trajectory_length):
            # Get current state
            current_state = pomdp.env.state
            traj_states.append(STATES.index(current_state))

            # Select random action (for data collection)
            action = random.choice(ACTIONS)
            action_idx = ACTIONS.index(action)
            traj_actions.append(action_idx)

            # Execute action and get observation
            reward = pomdp.env.state_transition(action, execute=True)
            traj_rewards.append(reward)

            # Get observation with noise
            observation = pomdp.agent.observation_model.sample(pomdp.env.state, action)
            noise = np.random.normal(0, 0.25, size=len(observation))
            noisy_observation = np.clip(observation + noise, 0, 1)
            traj_observations.append(noisy_observation)

        observations.append(np.array(traj_observations))
        actions.append(np.array(traj_actions))
        true_states.append(np.array(traj_states))
        rewards.append(np.array(traj_rewards))

    print(f"Generated {n_trajectories} trajectories of length {trajectory_length}")
    return pomdp, observations, actions, true_states, rewards


def create_continuous_medical_pomdp(init_state=None, trans_params=None, obs_params=None):
    if obs_params is None:
        obs_params = DEFAULT_OBS_PARAMS
    if trans_params is None:
        trans_params = DEFAULT_TRANS_PARAMS

    # Assicurati di utilizzare le versioni configurabili dei modelli
    transition_model = MedicalTransitionModel(trans_params)
    reward_model = MedicalRewardModel()
    obs_model = ContinuousObservationModel(STATES, ACTIONS, obs_params)
    policy_model = MedicalPolicyModel()

    init_belief = pomdp_py.Histogram({
        State("healthy"): 1 / 3,
        State("sick"): 1 / 3,
        State("critical"): 1 / 3
    })

    agent = Agent(
        init_belief=init_belief,
        policy_model=policy_model,
        transition_model=transition_model,
        observation_model=obs_model,
        reward_model=reward_model)

    init_state = init_state if init_state else random.choice(STATES).name

    env = Environment(init_state=State(init_state),
                      transition_model=transition_model,
                      reward_model=reward_model)
    return POMDP(agent, env)


def plot_observation_distribution(n_samples=5000):
    # Instantiate the continuous observation model
    obs_model = ContinuousObservationModel(STATES, ACTIONS)

    # Sample observations per state

    records = []
    for state in STATES:
        for _ in range(n_samples):
            test_val, symp_val = obs_model.sample(state, ACTIONS[0])
            records.append({"state": state.name, "test_result": test_val, "symptoms": symp_val})

    df = pd.DataFrame.from_records(records)

    # Plot 2D KDE for each state
    g = sns.FacetGrid(df, col="state", height=4, sharex=True, sharey=True)
    g.map_dataframe(
        sns.kdeplot,
        x="test_result",
        y="symptoms",
        fill=True,
        levels=6,
        cmap="viridis"
    )
    g.set_axis_labels("Test result", "Symptoms")
    g.tight_layout()
    plt.show()


def predict_next_observation_reward(fuzzy_model, current_observation, action):
    # Convert observation to test result and symptoms values
    action_val = fuzzy_model.action_mapping[action.name]
    test_val = current_observation[0]
    symptom_val = current_observation[1]

    # Pass values to fuzzy system
    fuzzy_model.prediction.input['action_input'] = action_val
    fuzzy_model.prediction.input['test_result'] = test_val
    fuzzy_model.prediction.input['symptoms'] = symptom_val

    # Compute result
    fuzzy_model.prediction.compute()

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

    return np.array([next_test_val, next_symptom_val]), reward_val


""" 
def evaluate_fuzzy_reward_prediction(trials=100, horizon=20):

    #Compares the reward predicted by the fuzzy model against the true reward
    #and evaluates next-observation prediction accuracy over simulated trajectories.


    np.random.seed(42)
    random.seed(42)

    base_reward_model = MedicalRewardModel()
    fuzzy_model = build_fuzzymodel()
    transition_model = MedicalTransitionModel()
    observation_model = ContinuousObservationModel(STATES, ACTIONS)

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
            true_next_obs.append(actual_obs)
            pred_next_obs.append(pred_obs)
            current_state = next_state

    # Convert to numpy arrays
    y_true = np.array(true_rewards)
    y_pred = np.array(fuzzy_predicted_rewards)
    test_true = np.array([obs[0] for obs in true_next_obs])
    test_pred = np.array([obs[0] for obs in pred_next_obs])
    symp_true = np.array([obs[1] for obs in true_next_obs])
    symp_pred = np.array([obs[1] for obs in pred_next_obs])

    # Reward metrics
    r2_test = 0.0
    r_test = 0.0
    r2_sym = 0.0
    r_sym = 0.0

    if np.var(y_true) == 0 or np.var(y_pred) == 0:
        r2_rew = 0.0 if np.var(y_true) == 0 else r2_score(y_true, y_pred)
        r_rew = 0.0
        print("Warning: Constant data encountered in rewards, metrics might be misleading.")
    else:
        r2_rew = r2_score(y_true, y_pred)
        r_rew = np.corrcoef(y_true, y_pred)[0, 1]
        r2_test = r2_score(test_true, test_pred)
        r_test = np.corrcoef(test_true, test_pred)[0, 1]
        r2_sym = r2_score(symp_true, symp_pred)
        r_sym = np.corrcoef(y_true, y_pred)[0, 1]

    # Observation prediction accuracy
    #obs_accuracy = np.mean([t == p for t, p in zip(true_next_obs, pred_next_obs)])

    # Print results
    print("\n--- Fuzzy Reward & Observation Prediction Performance ---")
    print(f"Total steps: {trials * horizon}")
    print(f"Reward R²: {r2_rew:.4f}, Pearson’s r: {r_rew:.4f}")
    print(f"Test Result R²: {r2_test:.4f}, Pearson’s r: {r_test:.4f}")
    print(f"Symptoms R²: {r2_sym:.4f}, Pearson’s r: {r_sym:.4f}")

    return r2_rew, r_rew, r2_test

"""

if __name__ == "__main__":
    # Create the POMDP
    pomdp = create_continuous_medical_pomdp(init_state="healthy")
    #pomdp.agent.observation_model.plot_observation_distribution()
    # Evaluate the fuzzy model's performance
    #evaluate_fuzzy_reward_prediction(trials=5, horizon=10)
