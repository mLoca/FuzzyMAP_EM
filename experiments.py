"""
For each setup of the JSON fill, we will run three different experiments:
 - fEM with fixed transition (few data and noisy observations)
 - fEM with fixed observation (few data and noisy observations)
 - fEM with 0 clue (few data and noisy observations)
"""
import numpy as np
from matplotlib import pyplot as plt

import utils
from fuzzy_EM import FuzzyPOMDP, evaluate_fuzzy_reward_prediction, \
    visualize_observation_distributions
from fuzzy_model import build_fuzzymodel
from fuzzy_model import create_continuous_medical_pomdp
from fuzzy_model import collect_data

from continouos_pomdp_example import STATES, generate_pomdp_data

SEED = 42
N_STATES = 3  # healthy, sick, critical
N_ACTION = 2  # wait, treat
OBS_DIM = 2  # test_result and symptoms (continuous)
DATA_CONFIG = ["NOISY"]  #TODO: add "FEW_DATA" to test with few data and normal

PARAMETERS_CONFIG = ["FIXED_TRANSITIONS", "FIXED_OBSERVATIONS", "ZERO_CLUE",
                     "FIXED_OBSERVATIONS_CRITICAL",
                     "FIXED_OBSERVATIONS_SICK"]


def run_experiment(trajectory_length=5, n_trajectories=5, noise_sd=0.05, fuzzy_model=None,
                   config=None,
                   transition_matrices=None,
                   observation_parameters=None, fig_string=None):
    """
    Run the few data experiment with fixed transition and observation models.
    """

    # Create the POMDP environment
    formatted_transition_matrices = None
    formatted_observation_parameters = None
    if transition_matrices is not None:
        formatted_transition_matrices = utils.from_matrix_to_triplelist(transition_matrices)
    if observation_parameters is not None:
        formatted_observation_parameters = utils.from_observation_config_to_observation(observation_parameters)

    pomdp_with_param = create_continuous_medical_pomdp(trans_params=formatted_transition_matrices,
                                                       obs_params=formatted_observation_parameters)

    # Collect data with few trials and horizon

    original_pomdp, observations, actions, true_states, rewards = generate_pomdp_data(
        trajectory_length=trajectory_length, n_trajectories=n_trajectories, seed=SEED, noise_sd=noise_sd,
        pomdp=pomdp_with_param
    )

    # to reuse the fuzzy model if it is already built
    if fuzzy_model is None:
        with utils.suppress_output():
            fuzzy_model = build_fuzzymodel(original_pomdp, seed=SEED)
        evaluate_fuzzy_reward_prediction(300, 10, fuzzy_model=fuzzy_model, pomdp=original_pomdp, seed=SEED)
        original_pomdp.agent.observation_model.plot_observation_distributions_2_axes()

    fit_and_performance(
        original_pomdp, observations, actions, config,
        fuzzy_model=fuzzy_model,
        fix_transitions=transition_matrices if config["fix_transitions"] else None,
        fix_observations=observation_parameters if config["fix_observations"] else None,
        fig_string=fig_string+"_Fuzzy.png")

    fit_and_performance(
        original_pomdp, observations, actions, config,
        fuzzy_model=None,
        fix_transitions=transition_matrices if config["fix_transitions"] else None,
        fix_observations=observation_parameters if config["fix_observations"] else None,
        fig_string=fig_string+"_Standard.png")

    return fuzzy_model


def fit_and_performance(original_pomdp, observations, actions, config, fuzzy_model=None,
                        fix_transitions=None, fix_observations=None, fig_string = None):
    """
    Fit the fuzzy POMDP and evaluate its performance.
    """
    if fuzzy_model is not None:
        use_fuzzy = True
        rho_T = config["rho_T"]
        rho_O = config["rho_O"]
    else:
        rho_T = 0.0
        rho_O = 0.0
        use_fuzzy = False

    fuzzy_pomdp = FuzzyPOMDP(n_states=N_STATES, n_actions=N_ACTION, obs_dim=OBS_DIM, use_fuzzy=use_fuzzy,
                             fuzzy_model=fuzzy_model, rho_T=rho_T, rho_O=rho_O, fix_transitions=fix_transitions,
                             fix_observations=fix_observations,
                             fixed_observation_states=config["fix_observations_states"])

    fuzzy_ll = fuzzy_pomdp.fit(
        observations, actions,
        max_iterations=200,
        tolerance=1e-5
    )

    np.set_printoptions(precision=2, formatter={'float': '{: 0.2f}'.format})
    print(f"Learning POMDP transitions:\n{fuzzy_pomdp.transitions}")


    #save the plot
    title_prefix = "Fuzzy" if use_fuzzy else "Standard"
    visualize_observation_distributions(fuzzy_pomdp, N_STATES, title_prefix=title_prefix)
    plt.show()
    if fig_string is not None:
        plt.savefig("img/"+fig_string)


    fuzzy_pomdp.compare_state_transitions(
        fuzzy_pomdp.transitions,
        original_pomdp.env.transition_model.transitions,
        original_pomdp.agent.observation_model,
        STATES
    )


def run_set_of_experiments():
    """
    Run a set of experiments with different configurations.
    """
    config = utils.get_experiment_config()

    for key, value_config in config.items():
        fuzzy_model_tmp = None
        for data_config in DATA_CONFIG:
            traj_length = 5
            n_trajectories = 20
            noise_sd = 0.0

            if data_config == "NORMAL":
                traj_length = 5
                n_trajectories = 20
                noise_sd = 0.02
            elif data_config == "NOISY":
                traj_length = 5
                n_trajectories = 30
                noise_sd = 0.15
            elif data_config == "FEW_DATA":
                traj_length = 3
                n_trajectories = 5
                noise_sd = 0.02

            for parameters_config in PARAMETERS_CONFIG:
                print("###")
                print(f"Running experiment {key} with {data_config} and {parameters_config}")
                print("###")
                transition_matrices = value_config["transitions"]
                observation_parameters = value_config["observations"]
                value_config["fix_observations_states"] = None

                if parameters_config == "FIXED_TRANSITIONS":
                    value_config["fix_transitions"] = True
                    value_config["fix_observations"] = False
                    value_config["fix_observations_states"] = None
                elif parameters_config == "FIXED_OBSERVATIONS":
                    value_config["fix_transitions"] = False
                    value_config["fix_observations"] = True
                    value_config["fix_observations_states"] = [0, 1, 2]
                elif parameters_config == "FIXED_OBSERVATIONS_CRITICAL":
                    value_config["fix_transitions"] = False
                    value_config["fix_observations"] = True
                    value_config["fix_observations_states"] = [2]
                elif parameters_config == "FIXED_OBSERVATIONS_SICK":
                    value_config["fix_transitions"] = False
                    value_config["fix_observations"] = True
                    value_config["fix_observations_states"] = [1]
                else:
                    value_config["fix_transitions"] = False
                    value_config["fix_observations"] = False
                    value_config["fix_observations_states"] = None

                fig_string = f"{key}_{data_config}_{parameters_config}"

                fuzzy_model_tmp = run_experiment(trajectory_length=traj_length,
                                                 n_trajectories=n_trajectories,
                                                 noise_sd=noise_sd,
                                                 config=value_config,
                                                 fuzzy_model=fuzzy_model_tmp,
                                                 transition_matrices=transition_matrices,
                                                 observation_parameters=observation_parameters,
                                                 fig_string=fig_string)


if __name__ == "__main__":
    run_set_of_experiments()
