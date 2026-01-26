import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

import numpy as np
import yaml
import itertools
import copy
from pomdp_py import Agent, Environment

import utils.utils as utils
from envs.continuous_medical_pomdp import ContinuousObservationModel

from models.trainable.pomdp_EM import PomdpEM
from models.trainable.pomdp_MAP_EM import PomdpMAPEM
from models.trainable.fuzzy_EM import FuzzyPOMDP
from fuzzy.fuzzy_model import build_fuzzymodel

from envs.medical_pomdp import *
from utils.metrics import compute_error_metrics, visualize_L1_trials, visualize_KL_trials, plot_grid_search_heatmap, \
    plot_1d_sensitivity

obs_index = {"test": 0, "symptoms": 1}


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
        self.distribution_type = distribution_type
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

    def _generate_POMDP(self, config):
        transition_model = MedicalTransitionModel(config["states"], self.true_transitions)
        obs_model = ContinuousObservationModel(config["states"], config["actions"], self.true_observations,
                                               distribution=self.distribution_type)
        # Reward and Policy models are needed for the Agent structure but not for data generation logic
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


def _expand_models_for_grid_search_(config_models):
    expanded_models = []

    for model_config in config_models:
        if not model_config.get("active", True):
            continue

        if not model_config.get("grid_search", False):
            expanded_models.append(model_config)
            continue

        param = model_config.get("params", {})
        model_name = model_config["name"]

        # Separate static and dynamic parameters
        static_params = {k: v for k, v in param.items() if not isinstance(v, list)}
        dynamic_params = {k: v for k, v in param.items() if isinstance(v, list)}

        if not dynamic_params:
            expanded_models.append(model_config)
            continue

        params_names = list(dynamic_params.keys())
        params_values = list(dynamic_params.values())
        params_combinations = list(itertools.product(*params_values))

        for combination in params_combinations:
            new_config = copy.deepcopy(model_config)
            new_config["grid_search"] = False

            combination_params = dict(zip(params_names, combination))
            new_config["params"] = {**static_params, **combination_params}

            # Give it a unique name
            param_suffix = "_".join([f"{k}={v}" for k, v in combination_params.items()])
            new_config["name"] = f"{model_name}_{param_suffix}"
            expanded_models.append(new_config)

    return expanded_models


def _instantiate_model_from_config(model_config, env, fuzzy_model, seed):
    """
    Instantiate a model based on the provided configuration.
    """
    model_cls = model_config["class"]
    model_params = model_config.get("params", {})

    common_args = {
        "n_states": env.n_states,
        "n_actions": env.n_actions,
        "obs_dim": env.obs_dim,
        "verbose": False,
        "seed": seed
    }

    if model_cls == "PomdpEM":
        model = PomdpEM(**common_args, **model_params)
    elif model_cls == "FuzzyPOMDP":
        model = FuzzyPOMDP(**common_args,
                           fuzzy_model=fuzzy_model,
                           obs_var_index=obs_index,
                           ensure_psd=True,
                           **model_params)
    elif model_cls == "PomdpMAPEM":
        model = PomdpMAPEM(**common_args, **model_params)
    else:
        raise ValueError(f"Unknown model class: {model_cls}")

    return model


def _check_cache(config_models, env_config, exp_id, data_size, noise_sd, trial, cache_dir,
                 use_cache, verbose):
    """
    Checks for existing cached results and returns models that need to be run.
    """
    cached_results = []
    models_to_run = []
    for model_config in config_models:
        if model_config.get("active", True):
            model_name = model_config["name"]
            model_name = model_name.replace("/", "_").replace(" ", "_").replace("=", "")

            # Safe filename generation
            filename = f"sz{data_size}_ns{noise_sd}_tr{trial}.pkl"
            file_path = f"{exp_id}/{env_config['name']}/{model_name}/"
            dir_path = os.path.join(cache_dir, file_path)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            filepath = f"{dir_path}/{filename}"

            if use_cache and os.path.exists(filepath):
                try:
                    with open(filepath, 'rb') as f:
                        cached_data = pickle.load(f)
                        cached_results.append(cached_data)
                    if verbose: print(f" [Cache] Loaded {filename}")
                except Exception:
                    models_to_run.append((model_config, filepath))
            else:
                models_to_run.append((model_config, filepath))

    return models_to_run, cached_results


def _train_and_evaluate_model(model_config, obs, acts, env, fuzzy_model, seed, standard_param):
    """Trains a single model and computes metrics."""
    model_name = model_config["name"]
    print(f" ... Training {model_name} ...")

    try:
        model = _instantiate_model_from_config(model_config, env, fuzzy_model, seed)
        start_time = time.time()

        fit_ll = model.fit(
            obs, acts,
            max_iterations=standard_param.get("n_iterations", 100),
            tolerance=float(standard_param.get("tolerance", 1e-4))
        )

        metrics = compute_error_metrics(
            model,
            env.original_transitions,
            env.original_observations,
            env.states,
            dist_type=env.distribution_type
        )

        elapsed_time = time.time() - start_time
        print(f" ... {model_name} finished in {elapsed_time:.2f}s. Final LL: {fit_ll:.2f}")

        return fit_ll, elapsed_time, metrics

    except Exception as e:
        print(f" [Error] Model {model_name} failed: {e}")
        return None


def run_dataset_batch(exp_id, trial, env_config, data_size, seq_length, noise_sd, seed, config_models, standard_param,
                      use_cache=True, verbose=False, cache_dir="res/cache/"):
    """
    Generates ONE dataset for this run_id/size.
    Evaluates ALL models in 'model_configs_list' on this exact dataset.
    """

    if not os.path.exists(cache_dir) and use_cache:
        os.makedirs(cache_dir, exist_ok=True)

    current_seed = seed + trial

    models_to_run, cached_results = _check_cache(config_models, env_config, exp_id, data_size, noise_sd, trial,
                                                 cache_dir, use_cache, verbose)
    if len(models_to_run) == 0:
        return cached_results

    # Generate dataset
    dist_type = env_config.get("distribution_type", "mvn")
    env = SyntheticEnvironment(env_config, distribution_type=dist_type)
    obs, acts = env.generate_data(data_size, seq_length, noise_sd, seed=current_seed)

    fuzzy_model = build_fuzzymodel(env.pomdp, seed=current_seed)

    np.random.seed(current_seed)
    random.seed(current_seed)
    # Train and evaluate each missing model
    results_batch = []
    for model_config, filepath in models_to_run:
        model_name = model_config["name"]

        if model_config["active"] is True:
            fit_ll, elapsed_time, metrics = _train_and_evaluate_model(model_config, obs, acts, env, fuzzy_model,
                                                                      current_seed,
                                                                      standard_param)

            result_entry = {
                "model_name": model_name,
                "env_name": env_config['name'],
                "data_size": data_size,
                "sequence_length": seq_length,
                "noise_sd": noise_sd,
                "trial": trial,
                "final_log_likelihood": fit_ll,
                "training_time_sec": elapsed_time,
                "metrics": metrics
            }
            results_batch.append(result_entry)

            print(f" Model {model_name} trained successfully in {elapsed_time:.2f} seconds.")
            print(f"  Final metrics: {metrics}")
            if use_cache:
                with open(filepath, 'wb') as f:
                    pickle.dump(result_entry, f)
                cached_results.append(results_batch)

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
    use_cache = global_settings.get("use_cache", True)
    all_environments = {env['name']: env for env in config["environments"]}
    results_summary = {}

    for exp_id, exp_config in config['experiments'].items():
        if not exp_config.get("active", True):
            continue

        print(f"Running experiment {exp_id}...")
        print(f"Experiment description: {exp_config.get('description', 'No description provided.')}")

        folder_name = exp_config.get("folder_name", "res/")
        config_models = _expand_models_for_grid_search_(exp_config["models"])

        results_summary[exp_id] = {'folder_name': folder_name}
        seq_length = exp_config["sequence_length"]
        n_trials = exp_config["n_trials"]
        noise_levels = exp_config["noise_level"]
        dataset_sizes = exp_config["dataset_sizes"]
        standard_params = exp_config.get("standard_params", {})

        tasks = []
        for env_name in exp_config["environments"]:
            if env_name not in all_environments:
                print(f"Environment {env_name} not found in configuration.")
                continue

            print(f"Using environment: {env_name}")
            env_config = all_environments[env_name]
            results_summary[exp_id][env_name] = {}

            # One task per (Dataset Size,  noise_level,  n_trial)
            # Each task will run ALL batch_configs on that specific dataset
            for data_size in dataset_sizes:
                for noise_sd in noise_levels:
                    for trial in range(n_trials):
                        tasks.append((exp_id, trial, env_config, data_size, seq_length, noise_sd, seed, config_models,
                                      standard_params, use_cache, verbose))

        all_batch_results = []
        if not (global_settings.get("parallel_execution", False)):
            for t in tasks:
                all_batch_results.append(run_dataset_batch(*t))
        else:
            with ProcessPoolExecutor(max_workers=min(os.cpu_count(), 32)) as executor:
                futures = [executor.submit(run_dataset_batch, *t) for t in tasks]
                for future in as_completed(futures):
                    try:
                        all_batch_results.append(future.result())
                    except Exception as e:
                        print(f"    [Batch Failed]: {e}")

        # Organize results
        for batch in all_batch_results:
            for res in batch:
                m_name = res.get('model_name', 'unknown')
                env_name = res.get('env_name', 'unknown')
                if m_name not in results_summary[exp_id][env_name]:
                    results_summary[exp_id][env_name][m_name] = []
                results_summary[exp_id][env_name][m_name].append(res)

    print("All experiments completed. Summary of results:")
    #TODO: Make it more elegant and general
    for exp_id, env_results in results_summary.items():
        print(f"Experiment {exp_id}:")
        folder_name = env_results.pop('folder_name', 'res/')
        for env_name, model_results in env_results.items():
            if "grid_search" in exp_id:
                plot_grid_search_heatmap(
                    model_results, metric_key='avg_l1_error',
                    title=f"Grid Search L1 Error", vmax=1.1,
                    exp_name=exp_id, folder_name="res/hyperparameter_grid_search/"
                )

                plot_grid_search_heatmap(
                    model_results, metric_key='final_kl',
                    title=f"Grid Search KL Divergence", vmax=7,
                    exp_name=exp_id, folder_name="res/hyperparameter_grid_search/"
                )

            elif "adaptive" in exp_id:
                plot_1d_sensitivity(
                    model_results,  param_name='alpha_ah', metric_key='avg_l1_error',
                    title=f"Adaptive Sensitivity: L1 Error", vmax=1.2, vmin=0.0,
                    folder_name="res/adaptive_alpha_search/"
                )
                plot_1d_sensitivity(
                    model_results, param_name='alpha_ah', metric_key='final_kl',
                    title=f"Adaptive Sensitivity: KL Divergence vs Alpha", vmax=12, vmin=0,
                    folder_name="res/adaptive_alpha_search/"
                )
            else:
                print(f" Environment: {env_name}")
                noise_levels = sorted({res['noise_sd'] for results in model_results.values() for res in results})

                for noise in noise_levels:
                    print(f"  Noise SD: {noise}")
                    filtered_results = {}
                    for model_name, results in model_results.items():
                        filtered = [r for r in results if r['noise_sd'] == noise]
                        if filtered:
                            filtered_results[model_name] = filtered

                    visualize_L1_trials(filtered_results, noise_level=noise, env_name=env_name, folder_name=folder_name)
                    visualize_KL_trials(filtered_results, noise_level=noise, env_name=env_name, folder_name=folder_name)

                    for model_name, results in filtered_results.items():
                        print(f"  Model: {model_name}")
                        for res in results:
                            print(f"   Data Size: {res['data_size']}, Seq Length: {res['sequence_length']}, "
                                  f"Noise SD: {res['noise_sd']}, Trial: {res['trial']}, "
                                  f"Final LL: {res['final_log_likelihood']:.2f}, "
                                  f"Training Time (s): {res['training_time_sec']:.2f}, "
                                  f"Metrics: {res['metrics']}")


if __name__ == "__main__":
    main()
