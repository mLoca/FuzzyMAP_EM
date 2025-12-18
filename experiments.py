"""
For each setup of the JSON fill, we will run three different experiments:
 - fEM with fixed transition (few data and noisy observations)
 - fEM with fixed observation (few data and noisy observations)
 - fEM with 0 clue (few data and noisy observations)
"""
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

import yaml

import numpy as np
from matplotlib import pyplot as plt
from pomdp_py import Agent, Environment

import utils.utils as utils

from models.trainable.pomdp_EM import PomdpEM
from models.trainable.fuzzy_EM import FuzzyPOMDP, evaluate_fuzzy_reward_prediction, \
    visualize_observation_distributions, _visualize_comparison_observation_distributions
from fuzzy.fuzzy_model import create_continuous_medical_pomdp, build_fuzzymodel

from continouos_pomdp_example import ContinuousObservationModel
from pomdp_example import *
from utils.metrics import compute_error_metrics


class SyntheticEnvironment:
    def __init__(self, config, distribution_type="mvn"):
        self.n_states = config['n_states']
        self.n_actions = config['n_actions']
        self.obs_dim = config['obs_dim']

        self.states = config['states']
        self.actions = config['actions']

        self.true_transitions = utils.from_matrix_to_triplelist(
            config['true_transitions'],
            states=config['states'],
            actions=config['actions']
        )
        if distribution_type == "mvn":
            self.true_observations = utils.from_observation_config_to_observation(
                config['true_observations'],
                states=config['states'],
                obs_names=["mean", "cov"]
            )
        elif distribution_type == "beta":
            self.true_observations = utils.from_observation_config_to_observation(
                config['true_observations'],
                states=config['states'],
                obs_names=config['observations']
            )

        # Store original for evaluation
        self.original_observations = config['true_observations']
        self.original_transitions = config['true_transitions']

        self.pomdp = self._generate_POMDP(config)

    def generate_data(self, data_size, seq_length, noise_sd, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        observations = []
        actions = []

        for _ in range(data_size):
            self._reset_environment()
            obs_seq = []
            act_seq = []
            current_state = self.pomdp.env.state
            for _ in range(seq_length):
                action = self.pomdp.agent.policy_model.sample(current_state)
                _ = self.pomdp.env.state_transition(MedAction(action), execute=True)
                current_state = self.pomdp.env.state
                observation = self.pomdp.agent.observation_model.sample(current_state, action)
                noise = np.random.normal(0, noise_sd, size=len(observation))
                noisy_observation = np.clip(observation + noise, 0, 1)

                action_idx = self.actions.index(action)
                obs_seq.append(noisy_observation)
                act_seq.append(action_idx)

            observations.append(obs_seq)
            actions.append(act_seq)

        return observations, actions

    def _generate_POMDP(self, config):
        transition_model = MedicalTransitionModel(config["states"], self.true_transitions)
        obs_model = ContinuousObservationModel(config["states"], config["actions"], self.true_observations,
                                               distribution="norm")
        policy_model = MedicalPolicyModel()

        init_belief = pomdp_py.Histogram({
            State("healthy"): 1 / 3,
            State("sick"): 1 / 3,
            State("critical"): 1 / 3
        })

        # NOTE: reward model is not used in data generation
        reward_model = MedicalRewardModel()

        agent = Agent(
            init_belief=init_belief,
            policy_model=policy_model,
            transition_model=transition_model,
            observation_model=obs_model,
            reward_model=reward_model)

        env = Environment(init_state=State(random.choice(config["states"])),
                          transition_model=transition_model,
                          reward_model=reward_model)
        return pomdp_py.POMDP(agent, env)

    def _reset_environment(self):
        """
        Create a new POMDP instance from an existing one.
        This is useful for resetting the environment.
        """
        init_belief = pomdp_py.Histogram({
            State("healthy"): 1 / 3,
            State("sick"): 1 / 3,
            State("critical"): 1 / 3
        })

        self.pomdp.agent.set_belief(init_belief, prior=True)
        self.pomdp.agent.tree = None


def run_dataset_batch(trial, env_config, data_size, seq_length, noise_sd, seed, config_models, standard_param, verbose):
    """
    Generates ONE dataset for this run_id/size.
    Evaluates ALL models in 'model_configs_list' on this exact dataset.
    """

    results_batch = []
    results_model = []

    # Generate dataset

    env = SyntheticEnvironment(env_config)
    obs, acts = env.generate_data(data_size, seq_length, noise_sd, seed=seed)

    fuzzy_model = build_fuzzymodel(env.pomdp,
                                   seed=seed)

    np.random.seed(seed)
    random.seed(seed)

    for model_config in config_models:
        model_name = model_config["name"]
        model_cls = model_config["class"]
        model_params = model_config.get("params", {})

        try:
            params = model_params.copy()

            #TODO: fixed part if is necessary.
            #if config_item.get('fix_transitions', False):
            #    params['fix_transitions'] = env.true_transitions
            #if config_item.get('fix_observations', False):
            #    pass
            #if 'fixed_observation_states' in config_item:
            #    params['fixed_observation_states'] = config_item['fixed_observation_states']
            if model_cls == "PomdpEM":
                model = PomdpEM(n_states=env.n_states,
                                n_actions=env.n_actions,
                                obs_dim=env.obs_dim,
                                verbose=verbose,
                                seed=seed,
                                **params)
            elif model_cls == "FuzzyPOMDP":

                model = FuzzyPOMDP(n_states=env.n_states,
                                   n_actions=env.n_actions,
                                   obs_dim=env.obs_dim,
                                   verbose=verbose,
                                   seed=seed,
                                   fuzzy_model=fuzzy_model,
                                   **params)
            else:
                raise ValueError(f"Unknown model class: {model_cls}")

            print(f"\\n Training model {model_name} on dataset (size={data_size}, seq_length={seq_length}, "
                  f"noise_sd={noise_sd}, trial={trial})...")

            start_time = time.time()
            #TODO: add the kmeans initialization if necessary

            fit_ll = model.fit(obs, acts,
                               max_iterations=standard_param["n_iterations"],
                               tolerance=float(standard_param["tolerance"]))
            #visualize_observation_distributions(model, env.n_states, title_prefix=f"{model_name}",
            #                                    datasize=f"{data_size}_noise{noise_sd}")
            metrics = compute_error_metrics(model,
                                            env.original_transitions,
                                            env.original_observations,
                                            env.states)

            end_time = time.time()
            elapsed_time = end_time - start_time
            results_model.append({
                "name": model_name,
                "model": model,
                "perm_ord": metrics['perm_ord']
            })

            results_batch.append({
                "model_name": model_name,
                "env_name": env_config['name'],
                "data_size": data_size,
                "sequence_length": seq_length,
                "noise_sd": noise_sd,
                "trial": trial,
                "final_log_likelihood": fit_ll,
                "training_time_sec": elapsed_time,
                "metrics": metrics
            })

            print(f" Model {model_name} trained successfully in {elapsed_time:.2f} seconds.")
            print(f"  Final metrics: {metrics}")

        except Exception as e:
            print(f" Model {model_name} failed: {e}")

    _visualize_comparison_observation_distributions(results_model, env.n_states,"",
                                                       f"{data_size} Noise: {noise_sd}")
    return results_batch

def main():
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'experiments.yaml')
    if not os.path.exists(config_path):
        print(f"Configuration file {config_path} not found.")
        return

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    global_settings = config["global_settings"]
    seed = global_settings.get("seed", 42)
    verbose = global_settings.get("verbose", True)
    environments = config["environments"]
    all_env_names = [env['name'] for env in environments]
    results_summary = {}

    for exp_id, exp_config in config['experiments'].items():
        if not exp_config.get("active", True):
            continue

        print(f"Running experiment {exp_id}...")
        print(f"Experiment description: {exp_config.get('description', 'No description provided.')}")

        results_summary[exp_id] = {}
        seq_length, n_trials, noise_levels = exp_config["sequence_length"], exp_config["n_trials"], exp_config[
            "noise_level"]
        env_names, dataset_sizes = exp_config["environments"], exp_config["dataset_sizes"]

        tasks = []
        for env_name in env_names:
            if env_name not in all_env_names:
                print(f"Environment {env_name} not found in configuration.")
                continue

            print(f"Using environment: {env_name}")
            env_config = [e for e in environments if e['name'] == env_name][0]
            results_summary[exp_id][env_name] = {}

            config_models = exp_config["models"]
            standard_param = exp_config.get("standard_params", {})
            # One task per (Dataset Size,  noise_level,  n_trial)
            # Each task will run ALL batch_configs on that specific dataset

            for data_size in dataset_sizes:
                for noise_sd in noise_levels:
                    for trial in range(n_trials):
                        tasks.append((trial, env_config, data_size, seq_length, noise_sd, seed, config_models,
                                      standard_param, verbose))

        if not(global_settings.get("parallel_execution", False)):
            # Run tasks SEQUENTIALLY
            for t in tasks:
                batch_res = run_dataset_batch(*t)
                # Unpack batch results into the structured summary
                for res in batch_res:
                    m_name = res.get('model_name', 'unknown')
                    env_name = res.get('env_name', 'unknown')
                    if m_name not in results_summary[exp_id][env_name]:
                        results_summary[exp_id][env_name][m_name] = []
                    results_summary[exp_id][env_name][m_name].append(res)
        else:
            with ProcessPoolExecutor(max_workers=min(os.cpu_count(), 6)) as executor:
                futures = [executor.submit(run_dataset_batch, *t) for t in tasks]
                for future in as_completed(futures):
                    try:
                        batch_res = future.result()
                        # Unpack batch results into the structured summary
                        for res in batch_res:
                            m_name = res.get('model_name', 'unknown')
                            env_name = res.get('env_name', 'unknown')
                            if m_name not in results_summary[exp_id][env_name]:
                                results_summary[exp_id][env_name][m_name] = []
                            results_summary[exp_id][env_name][m_name].append(res)
                    except Exception as e:
                        print(f"    Batch failed: {e}")


    print("All experiments completed. Summary of results:")
    for exp_id, env_results in results_summary.items():
        print(f"Experiment {exp_id}:")
        for env_name, model_results in env_results.items():
            print(f" Environment: {env_name}")
            for model_name, results in model_results.items():
                print(f"  Model: {model_name}")
                for res in results:
                    print(f"   Data Size: {res['data_size']}, Seq Length: {res['sequence_length']}, "
                          f"Noise SD: {res['noise_sd']}, Trial: {res['trial']}, "
                          f"Final LL: {res['final_log_likelihood']:.2f}, "
                          f"Training Time (s): {res['training_time_sec']:.2f}, "
                          f"Metrics: {res['metrics']}")

if __name__ == "__main__":
    main()
