import random

import pomdp_py
from pomdp_py import ObservationModel, Agent, Environment, POMDP
import numpy as np
from scipy.stats import beta, norm, uniform
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from pomdp_example import State, MedAction, MedicalTransitionModel, MedicalRewardModel, \
    MedicalPolicyModel

STATES = [State(s) for s in ["healthy", "sick", "critical"]]
ACTIONS = [MedAction(a) for a in ["wait", "treat"]]
DEFAULT_OBS_PARAMS = {
    # Sano: Test per lo più negativo, sintomi per lo più assenti
    "healthy":{
        "test":(1.8, 7.5),
        "symptoms":(2, 10),
    },
    # Malato: Test misto ma per lo più negativo, sintomi per lo più presenti
    "sick": {
        "test":(2.5, 4.5),
        "symptoms":(4, 4.5),
    },
    # Critico: Test per lo più positivo, sintomi fortemente presenti
    "critical": {
        "test":(9, 2,),
        "symptoms":(8.9, 2.5),
    }
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
    _DIST_MAP = ["beta", "norm", "uniform"]

    def __init__(self, states, actions, obs_params=None, dist_name="beta", *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.states = states
        self.actions = actions

        if dist_name not in self._DIST_MAP:
            raise ValueError(f"Unsupported distribution type: {dist_name}. "
                             f"Supported types are: {self._DIST_MAP}")
        self.dist_name = dist_name
        # define  distribution parameters per state
        # format: { state_name: (alpha_test, beta_test, alpha_symp, beta_symp) }
        if obs_params is None:
            self.params = DEFAULT_OBS_PARAMS
        else:
            for state in states:
                if state.name not in obs_params:
                    raise ValueError(f"Observation parameters missing for state: {state.name}")
                for dim, p in obs_params[state.name].items():
                    if len(p) != 2:
                        raise ValueError(f"Beta distribution requires 2 parameters for {dim} in state {state.name}")

            self.params = obs_params

        #TODO: check the dimension of obs_params for each state

    def sample(self, next_state, action, **kwargs):
        dims = self.params[next_state.name]
        samples = []
        for _, p in dims:
            if self.dist_name == "beta":
                a, b = p
                samples.append(beta.rvs(a, b))
            elif self.dist_name == "norm":
                loc, scale = p
                samples.append(norm.rvs(loc=loc, scale=scale))
            elif self.dist_name == "uniform":
                low, high = p
                samples.append(uniform.rvs(loc=low, scale=(high - low)))
            else:
                raise ValueError(f"Distribuzione non supportata: {self.dist_name}")
        return np.asarray(samples)

    def probability(self, observation, next_state, action):
        obs = np.asarray(observation)
        dims = self.params[next_state.name]
        prob = 1.0
        for i, (_, p) in enumerate(dims):
            x = obs[i]
            if self.dist_name == "beta":
                a, b = p
                prob *= beta.pdf(x, a, b)
            elif self.dist_name == "norm":
                loc, scale = p
                prob *= norm.pdf(x, loc=loc, scale=scale)
            elif self.dist_name == "uniform":
                low, high = p
                prob *= uniform.pdf(x, loc=low, scale=(high - low))
            else:
                raise ValueError(f"Distribuzione non supportata: {self.dist_name}")
        return prob

    #TODO: move to a file responsible to plotting
    #TODO: make it general
    def plot_observation_distribution(self):
        # Plot the observation distribution for each state
        # Create a grid of subplots (2 rows: tests and symptoms, 3 columns: states)
        fig, axs = plt.subplots(len(self.params[self.states[0].name]), len(self.states), figsize=(15, 10))
        plt.suptitle("Observation Distribution for Each State", fontsize=16)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)

        # Loop through each state and plot the distributions
        cmap = self._get_colors()
        colors = [cmap(i / max(1, len(self.states) - 1)) for i in range(len(self.states))]

        for col, state in enumerate(self.states):
            axs[0, col].set_title(f"{state_name}")

            state_name = state.name
            index = 0
            for dim, p in self.params[state_name]:
                _obs_x = np.linspace(-5, 5, 5000)
                _obs_y = None
                if self.dist_name == "beta":
                    a, b = p
                    _obs_y = beta.pdf(_obs_x, a, b)
                elif self.dist_name == "norm":
                    loc, scale = p
                    _obs_y = norm.pdf(_obs_x, loc, scale)
                elif self.dist_name == "uniform":
                    low, high = p
                    _obs_y = uniform.pdf(_obs_x, low, high)

                # Plot test results distribution (first row)
                axs[index, col].plot(_obs_x, _obs_y, color=colors[col])
                axs[index, col].set_title(f"{state_name}")
                axs[index, col].set_xlabel(dim)
                axs[index, col].set_ylabel("Density")

                index += 1

                # Plot symptoms distribution (second row)

        plt.tight_layout()
        plt.show()

        a_t, b_t, a_s, b_s = self.params["sick"]
        x_test = np.linspace(0, 1, 100)
        y_test = beta.pdf(x_test, a_t, b_t)
        plt.plot(x_test, y_test, label='Test Result', color='blue')
        plt.savefig("sick_test_distribution.png")

    def _get_colors(self):
        num_states = len(self.states)
        if num_states <= 10:
            cmap = plt.get_cmap('tab10')
        elif num_states <= 20:
            cmap = plt.get_cmap('tab20')
        else:
            cmap = plt.get_cmap('viridis')
        return cmap

    def plot_observation_distributions_2_axes(self, obs_names):
        """
        Plot 2D joint probability density functions for test results and symptoms
        for each state using contour plots.

        obs_names: List of observation names to plot (e.g., ["test", "symptoms"]). HAS TO BE A 2D OBSERVATION
        """
        for obs_name in obs_names:
            if obs_name not in self.params[self.states[0].name]:
                raise ValueError(f"Unsupported observation name: {obs_name}. "
                                 f"Supported names are: {self.params[self.states[0].name].keys()}")
        fig, axes = plt.subplots(1, len(self.params[self.states[0].name].keys()), figsize=(18, 6))
        state_names = ["healthy", "sick", "critical"]

        # Create meshgrid for evaluation
        _obs_x0 = np.linspace(0, 1, 100)
        _obs_x1 = np.linspace(0, 1, 100)
        X0, X1 = np.meshgrid(_obs_x0, _obs_x1)

        joint_pdf = 1
        for i, state_name in enumerate(state_names):
            for dim, p in self.params[state_name]:
                if dim == obs_names[0] or dim == obs_names[1]:
                    if dim == obs_names[0]:
                        _obs_X = X0
                    else:
                        _obs_X = X1

                    if self.dist_name == "beta":
                        a, b = p
                        _obs_y = beta.pdf(_obs_X, a, b)
                    elif self.dist_name == "norm":
                        loc, scale = p
                        _obs_y = norm.pdf(_obs_X, loc, scale)
                    elif self.dist_name == "uniform":
                        low, high = p
                        _obs_y = uniform.pdf(_obs_X, low, high)

            joint_pdf *= _obs_y

            # Create contour plot
            im = axes[i].contourf(X0, X1, joint_pdf, levels=20, cmap='viridis')
            axes[i].set_title(f"State {state_name}")
            axes[i].set_xlabel(obs_names[0])
            axes[i].set_ylabel(obs_names[1])

            # Add colorbar
            cbar = fig.colorbar(im, ax=axes[i], pad=0.01)

        plt.tight_layout()
        plt.show()

#TODO: change the socpe
def generate_pomdp_data(n_trajectories, trajectory_length, noise_sd=0.01, seed=None, pomdp=None):
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
