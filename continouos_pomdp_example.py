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
    "sick": (2.5, 4.5, 4, 4.5),
    # Critico: Test per lo più positivo, sintomi fortemente presenti
    "critical": (9, 2, 8.9, 2.5)
}
DEFAULT_TRANS_PARAMS = {
    # Wait action transitions
    ('wait', 'healthy', 'healthy'): 0.85,
    ('wait', 'healthy', 'sick'): 0.14,
    ('wait', 'healthy', 'critical'): 0.01,
    ('wait', 'sick', 'healthy'): 0.3,
    ('wait', 'sick', 'sick'): 0.5,
    ('wait', 'sick', 'critical'): 0.2,
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
    ('treat', 'critical', 'sick'): 0.6,
    ('treat', 'critical', 'critical'): 0.3,
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

    def plot_observation_distributions_2_axes(self):
        """
        Plot 2D joint probability density functions for test results and symptoms
        for each state using contour plots.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        state_names = ["healthy", "sick", "critical"]

        # Create meshgrid for evaluation
        test_vals = np.linspace(0, 1, 100)
        symp_vals = np.linspace(0, 1, 100)
        T, S = np.meshgrid(test_vals, symp_vals)

        for i, state_name in enumerate(state_names):
            a_t, b_t, a_s, b_s = self.params[state_name]

            # Calculate joint PDF (product of individual PDFs)
            test_pdf = beta.pdf(T, a_t, b_t)
            symp_pdf = beta.pdf(S, a_s, b_s)
            joint_pdf = test_pdf * symp_pdf

            # Create contour plot
            im = axes[i].contourf(T, S, joint_pdf, levels=20, cmap='viridis')
            axes[i].set_title(f"Fuzzy State {i + 1}")
            axes[i].set_xlabel("Test Result")
            axes[i].set_ylabel("Symptoms")

            # Add colorbar
            cbar = fig.colorbar(im, ax=axes[i], pad=0.01)

        plt.tight_layout()
        plt.show()
    def plot_observation_distributions(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        for state in self.states:
            a_t, b_t, a_s, b_s = self.params[state.name]
            x_test = np.linspace(0, 1, 100)
            y_test = beta.pdf(x_test, a_t, b_t)
            ax.plot(x_test, y_test, label=f"{state.name} - Test Result")

            x_symp = np.linspace(0, 1, 100)
            y_symp = beta.pdf(x_symp, a_s, b_s)
            ax.plot(x_symp, y_symp, linestyle='--', label=f"{state.name} - Symptoms")

        ax.set_title("Observation Distribution")
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend()
        plt.show()

def generate_pomdp_data(n_trajectories, trajectory_length, noise_sd = 0.01, seed=None , pomdp=None):
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
        if pomdp is None:
            pomdp = create_continuous_medical_pomdp()
        else:
            pomdp = reset_pomdp(pomdp)
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
            noise = np.random.normal(0, noise_sd, size=len(observation))
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

def reset_pomdp(pomdp):
    """
    Create a new POMDP instance from an existing one.
    This is useful for resetting the environment without losing the agent's configuration.
    """
    init_belief = pomdp_py.Histogram({
        State("healthy"): 1 / 3,
        State("sick"): 1 / 3,
        State("critical"): 1 / 3
    })

    pomdp.agent.set_belief(init_belief, prior=True)
    pomdp.agent.tree = None

    return POMDP(
        agent=pomdp.agent,
        env=pomdp.env
    )

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


if __name__ == "__main__":
    # Create the POMDP
    pomdp = create_continuous_medical_pomdp(init_state="healthy")
    pomdp.agent.observation_model.plot_observation_distribution()
    #pomdp.agent.observation_model.plot_observation_distribution()
    # Evaluate the fuzzy model's performance
    #evaluate_fuzzy_reward_prediction(trials=5, horizon=10)
