import argparse
import mlflow
import torch
import numpy as np
import logging
import scipy.stats
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, parallel_backend
import random
from fuzzy.HIV_fuzzy_new import HIVExpertNewModel
from fuzzy.hiv_fuzzy import HIVExpert5DModel

# Import the custom simulator instead of whynot
from hiv_simulator import HIVSimulator
from utils.utils import my_hiv_reward_fn

# Import your existing models and metrics
from models.trainable.pomdp_EM import PomdpEM as POMDP_EM
from models.trainable.fuzzy_EM import FuzzyPOMDP as FuzzyMAP_EM
from utils.metrics import compute_avg_l1_error, compute_log_likelihood
from pathlib import Path

# ==============================================================================
# yaacovpariente/POMDPPlanners INTEGRATION
# ==============================================================================

from POMDPPlanners.planners.mcts_planners.pft_dpw import PFT_DPW
from POMDPPlanners.planners.mcts_planners.pomcp_dpw import POMCP_DPW
from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
from POMDPPlanners.planners.mcts_planners.pomcpow import POMCPOW
from POMDPPlanners.simulations.episodes import run_episode
from POMDPPlanners.utils.action_samplers import DiscreteActionSampler
from POMDPPlanners.configs.environment_configs import EnvironmentConfigsAPI
from POMDPPlanners.configs.planners_hyperparam_configs import PlannersHyperparamConfigs
from POMDPPlanners.core.simulation import (
    NumericalHyperParameter, CategoricalHyperParameter
)
from POMDPPlanners.core.simulation.hyperparameter_tuning import (
    HyperParameterRunParams, HyperParameterOptimizationDirection,  HyperParamPlannerConfig
)

from POMDPPlanners.core.belief import get_initial_belief
from POMDPPlanners.utils.belief_factory import create_environment_belief
from POMDPPlanners.simulations.simulation_apis.local_simulations_api import LocalSimulationsAPI
from POMDPPlanners.core.simulation import EnvironmentRunParams
from POMDPPlanners.utils.logger import get_logger

POMDPPLANNERS_AVAILABLE = True
logging.disable(logging.INFO)
#except ImportError:
#    POMDPPLANNERS_AVAILABLE = False
#    print("Warning: POMDPPlanners not found. Planning benchmark will be skipped unless installed.")
from envs.custom_env_to_plan import LearnedHIVEnvironment
from envs.true_hiv_pomdp import TrueHIVEnvironment

hiv_action_mapping = {
    0: {'e1': 0.0, 'e2': 0.0},   # None
    1: {'e1': 0.7, 'e2': 0.0},   # RTI Only
    2: {'e1': 0.0, 'e2': 0.3},   # PI Only
    3: {'e1': 0.7, 'e2': 0.3}    # HAART (Both)
}

class TrueHIVEnvironmentWrapper:
    """ 
    Wraps the HIVSimulator to conform to the POMDPPlanners Environment generative interface.
    The evaluator uses this to step the TRUE biological reality.
    """
    def __init__(self, hiv_sim):
        self.env = hiv_sim
        self.discount_factor = 0.95
        self.name = "TrueHIVEnvironment"

    def get_actions(self):
        return list(range(self.env.num_actions))

    def sample_initial_state(self):
        return self.env.reset(perturb_params=True)

    def step(self, state, action):
        # Override the simulator's internal state to ensure stateless MCTS simulations
        self.env.state = state
        next_state, reward, done, info = self.env.step(action)
        
        # Emulate the clinical measurement noise present in the training set
        obs = next_state + np.random.normal(0, 0.05, size=next_state.shape)
        return next_state, tuple(obs), float(reward), done, info

    def observation_probability(self, action, next_state, observation):
        """ Needed for particle filtering if the True env is queried for density """
        return scipy.stats.multivariate_normal.pdf(
            observation, 
            mean=next_state, 
            cov=np.eye(len(next_state)) * 0.05**2
        )

def evaluate_cross_environment(true_env, policy, initial_belief, episode = 1, max_steps=40, trial=0):
    """
    Evaluates a policy trained on a Learned Environment inside a True Environment.
    """
    # 1. The Body: Initialize true biological reality (6D array)
    discount_factor = true_env.discount_factor
    initial_state = true_env.initial_state_dist().sample()[0]
    seed = trial * 100 + episode
    np.random.seed(seed)
    random.seed(seed)
    #random_noise = np.random.uniform(-0.01, 0.01, size=6)
    #random_noise = np.random.uniform(-0.2, 0.01, size=4)
    #random_noise = np.concatenate((random_noise, np.random.uniform(-0.01, 0.2, size=1)))
    #random_noise = np.concatenate((random_noise, np.random.uniform(-0.2, 0.01, size=1)))
    random_noise = np.array([1,1,1,1,-1,1]) * np.random.uniform(0, 0.2, size=1)
    true_env = TrueHIVEnvironment(initial_biological_state=initial_state, discount_factor=discount_factor, p_init=random_noise)
    true_env.logger.disabled = True
    true_state = true_env.initial_state_dist().sample()[0]
    

    obs_0 = true_env.sample_observation(true_state, action=0)  # action=0 is a dummy
    
    # B. Extract the model's learned observation parameters (mu and Sigma)
    model_env = policy.environment  # This is std_env or fuzzy_env (LearnedHIVEnvironment)
    
    # C. Compute the likelihood of obs_0 under each of the 5 abstract states
    likelihoods = []
    for s in range(model_env.n_states):
        mean = model_env.mu[s]
        cov = model_env.Sigma[s]
        # Add a tiny epsilon to the diagonal of the covariance matrix for numerical stability
        cov_safe = cov + np.eye(len(mean)) * 1e-6
        likelihood = scipy.stats.multivariate_normal.pdf(obs_0, mean=mean, cov=cov_safe)
        likelihoods.append(likelihood)
        
    best_state = np.argmax(likelihoods)
    
    # D. Create a WeightedParticleBelief where all particles are initialized to the best_state (probability = 1.0)
    from POMDPPlanners.core.belief import WeightedParticleBelief
    n_particles = 100
    particles = np.full(n_particles, best_state, dtype=int)
    log_weights = np.log(np.ones(n_particles) / n_particles)
    belief = WeightedParticleBelief(particles=particles, log_weights=log_weights, resampling=True)
    
    total_reward = 0.0
    
    states_history = [true_state]
    rewards_history = []
    
    for step in range(max_steps):
        try:
            action = policy.plan(belief)
        except AttributeError:
            action = policy.action(belief)
        
        if step == 0:
            action = 0
        else:
            action = action[0][0]

        next_true_state = true_env.sample_next_state(true_state, action)

        obs = true_env.sample_observation(next_true_state, action)
        
        reward = true_env.reward(true_state, action, next_true_state)
        total_reward += reward
        
        states_history.append(next_true_state)
        rewards_history.append(reward)
        
        if true_env.is_terminal(next_true_state):
            break
            

        belief.update(action, obs, pomdp=policy.environment)
            
        # Move time forward
        true_state = next_true_state
        
    return total_reward, np.array(states_history), np.array(rewards_history)

def evaluate_planning_performance(true_env, em_model, fuzzy_model, n_episodes=50, horizon=15, hyper_optimize=False, trial=0):
    std_params = {
        "T": em_model.transitions,          
        "mu": em_model.obs_means,     
        "Sigma": em_model.obs_covs    
    }

    fuzzy_params = {
        "T": fuzzy_model.transitions,
        "mu": fuzzy_model.obs_means,
        "Sigma": fuzzy_model.obs_covs
    }
    discount_factor = 0.99

    std_env = LearnedHIVEnvironment("std_env", std_params, my_hiv_reward_fn, discount_factor=discount_factor) 
    fuzzy_env = LearnedHIVEnvironment("fuzzy_env", fuzzy_params, my_hiv_reward_fn, discount_factor=discount_factor)

    std_sampler = DiscreteActionSampler(std_env.get_actions())
    fuzzy_sampler = DiscreteActionSampler(fuzzy_env.get_actions())

    unhealthy_steady_state = [163573., 5., 11945., 46., 63919., 24.]
    true_env = TrueHIVEnvironment(initial_biological_state=unhealthy_steady_state, discount_factor=discount_factor)

    # 2. Configure the planners
    # Note: You need to tune these hyperparameters based on your HIV benchmark
    planner_config_std = {
        "n_simulations":1000,
        "depth": 3,
        "discount_factor": 0.99,
        "exploration_constant": 5000,
        "k_o": 4.868167504611893,
        "k_a": 4.203111794128549,
        "alpha_o": 0.3982346933640448,
        "alpha_a": 0.46669636595271263,
    }

    std_policy = POMCPOW(std_env,action_sampler=fuzzy_sampler, name="POMCP_Standard", **planner_config_std)
    planner_config_fuzzy = {
        "n_simulations":1000,
        "depth": 3,
        "discount_factor": 0.99,
        "exploration_constant": 15000,
        "k_o": 4.571573457147612,
        "k_a": 3.2061026430432484,
        "alpha_o": 0.21689427930779825,
        "alpha_a": 0.48918164871267744,
    }

    fuzzy_policy = POMCPOW(fuzzy_env,action_sampler=fuzzy_sampler, name="POMCP_Fuzzy", **planner_config_std)

    # 3. Setup Initial Beliefs (e.g., Uniform particle belief)
    # You can customize this based on the POMDPPlanners documentation
    std_initial_belief = get_initial_belief(std_env, n_particles=100)
    #fuzzy_initial_belief = get_initial_belief(fuzzy_env, n_particles=100)
    fuzzy_initial_belief = std_initial_belief  # For simplicity, using the same initial belief for both

    # 4. Run the Evaluation
    if hyper_optimize:
        print("Starting Hyperparameter Optimization for POMCPOW...")
        api = LocalSimulationsAPI(
            cache_dir_path=Path("./hyperparameter_results"),
            debug=True)
        # Hyperparamatters optimization
        #[fuzzy_env, std_env]
        for env in [std_env]:
            initial_belief = std_initial_belief if env.name == "std_env" else fuzzy_initial_belief
            optimization_configs = [
                HyperParameterRunParams(
                    environment=env,
                    belief=initial_belief,
                    hyper_param_planner_config=HyperParamPlannerConfig(
                        policy_cls=POMCPOW,
                        hyper_parameters=[
                            NumericalHyperParameter(0.1, 100., "exploration_constant"),  # Reduced range
                            NumericalHyperParameter(2, 4, "depth"),  # Reduced depth
                            NumericalHyperParameter(1.0, 5.0, "k_o"),  # Observation progressive widening coefficient
                            NumericalHyperParameter(1.0, 5.0, "k_a"),  # Action progressive widening coefficient
                            NumericalHyperParameter(0.01, 0.5, "alpha_o"),  # Observation progressive widening exponent
                            NumericalHyperParameter(0.01, 0.5, "alpha_a")   # Action progressive widening exponent
                        ],
                        constant_parameters={
                            "discount_factor": 0.99,
                            "n_simulations": 2000,  # Minimal simulations for testing
                            "action_sampler": std_sampler,
                            "name": "OptimizedPOMCPOW_HIV"
                        },
                    ),
                    num_episodes=25,       # Episodes for final evaluation
                    num_steps=50,          # Steps per episode
                    n_trials=25,         # Number of optimization trials
                    parameters_to_optimize=[("average_return", HyperParameterOptimizationDirection.MAXIMIZE)]
                )
            ]
            import os
            os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
            import mlflow
            results = api.run_hyperparameter_optimization(
                environment_run_params=optimization_configs,
                experiment_name="HIV_POMCP_Optimization",
                n_jobs=-1,  # Use all available CPU cores
            )
            #Analyze results
            print(f"Results for environment: {env.name}")
            for i, result in enumerate(results):
                print(f"Configuration {i+1} Results:")
                print(f"  Environment: {result.environment.__class__.__name__}")
                print(f"  Policy: {result.policy.__class__.__name__}")
                print(f"  Best hyperparameters: {result.chosen_hyper_parameters}")
                print(f"  Policy name: {result.policy.name}")

    print("Evaluating Standard and Fuzzy-MAP EM Models")
    
    jobs = []
    for ep in range(30):  
        jobs.append((std_policy, std_initial_belief, ep))
        jobs.append((fuzzy_policy, fuzzy_initial_belief, ep))

    all_results = Parallel(n_jobs=-1)(
        delayed(evaluate_cross_environment)(true_env, pol, belief, episode=ep, trial=trial) 
        for pol, belief, ep in jobs
    )
    
    # joblib.Parallel preserves the input order
    std_results = all_results[0::2]
    fuzzy_results = all_results[1::2]
    
    std_returns = [r[0] for r in std_results]
    std_states = [r[1] for r in std_results]
    std_rewards = [r[2] for r in std_results]

    fuzzy_returns = [r[0] for r in fuzzy_results]
    fuzzy_states = [r[1] for r in fuzzy_results]
    fuzzy_rewards = [r[2] for r in fuzzy_results]

    print("\n Performance Results:")
    print(f"Standard: {np.mean(std_returns)}.    Fuzzy:{np.mean(fuzzy_returns)}")
    print(f"Standard 95% CI: [{scipy.stats.norm.interval(0.95, loc=np.mean(std_returns), scale=scipy.stats.sem(std_returns))}]")
    print(f"Fuzzy 95% CI: [{scipy.stats.norm.interval(0.95, loc=np.mean(fuzzy_returns), scale=scipy.stats.sem(fuzzy_returns))}]")

    plot_patient_trajectories(std_states, std_rewards, fuzzy_states, fuzzy_rewards)
    return std_returns, fuzzy_returns

def plot_patient_trajectories(std_states, std_rewards, fuzzy_states, fuzzy_rewards, save_path='res/patient_trajectories.png'):
    import matplotlib.pyplot as plt
    import numpy as np

    def pad_arrays(arrays):
        max_len = max(len(arr) for arr in arrays)
        padded = []
        for arr in arrays:
            if len(arr) < max_len:
                pad_width = [(0, max_len - len(arr))] + [(0, 0)] * (arr.ndim - 1)
                padded.append(np.pad(arr, pad_width, mode='constant', constant_values=np.nan))
            else:
                padded.append(arr)
        return np.array(padded)
        
    std_states_padded = pad_arrays(std_states)
    fuzzy_states_padded = pad_arrays(fuzzy_states)
    std_rewards_padded = pad_arrays(std_rewards)
    fuzzy_rewards_padded = pad_arrays(fuzzy_rewards)
    
    # Convert states to log10 space to match the simulator's logspace setting
    std_states_padded = np.log10(np.clip(std_states_padded, 1e-10, None))
    fuzzy_states_padded = np.log10(np.clip(fuzzy_states_padded, 1e-10, None))
    
    import os
    if not os.path.exists('res'):
        os.makedirs('res')

    # Save the results to an npz file
    np.savez('res/patient_trajectories.npz', 
             std_states=std_states_padded, std_rewards=std_rewards_padded,
             fuzzy_states=fuzzy_states_padded, fuzzy_rewards=fuzzy_rewards_padded)
    print("Patient trajectories saved to res/patient_trajectories.npz")
    
    fig, axes = plt.subplots(4, 2, figsize=(15, 12))
    axes = axes.flatten()
    state_labels = ["T1 (Uninfected CD4)", "T2 (Infected CD4)", "T1* (Uninfected Macrophages)", "T2* (Infected Macrophages)", "V (Free Virus)", "E (Immune Response)"]
    
    for i in range(6):
        with np.errstate(all='ignore'):
            std_mean = np.nanmean(std_states_padded[:, :, i], axis=0)
            std_std = np.nanstd(std_states_padded[:, :, i], axis=0)
            fuz_mean = np.nanmean(fuzzy_states_padded[:, :, i], axis=0)
            fuz_std = np.nanstd(fuzzy_states_padded[:, :, i], axis=0)
        
        days = np.arange(len(std_mean)) * 5
        axes[i].plot(days, std_mean, label='Standard EM', color='blue')
        axes[i].fill_between(days, std_mean - std_std, std_mean + std_std, alpha=0.2, color='blue')
        
        axes[i].plot(days, fuz_mean, label='Fuzzy-MAP EM', color='orange')
        axes[i].fill_between(days, fuz_mean - fuz_std, fuz_mean + fuz_std, alpha=0.2, color='orange')
        
        axes[i].set_title(state_labels[i])
        axes[i].set_xlabel('Days')
        axes[i].set_ylabel('Log10(Cell Count / Virus Load)')
        if i == 0:
            axes[i].legend()
            
    with np.errstate(all='ignore'):
        std_r_mean = np.nanmean(std_rewards_padded, axis=0)
        std_r_std = np.nanstd(std_rewards_padded, axis=0)
        fuz_r_mean = np.nanmean(fuzzy_rewards_padded, axis=0)
        fuz_r_std = np.nanstd(fuzzy_rewards_padded, axis=0)
    
    days_r = np.arange(len(std_r_mean)) * 5
    axes[6].plot(days_r, std_r_mean, label='Standard EM', color='blue')
    axes[6].fill_between(days_r, std_r_mean - std_r_std, std_r_mean + std_r_std, alpha=0.2, color='blue')
    
    axes[6].plot(days_r, fuz_r_mean, label='Fuzzy-MAP EM', color='orange')
    axes[6].fill_between(days_r, fuz_r_mean - fuz_r_std, fuz_r_mean + fuz_r_std, alpha=0.2, color='orange')
    
    axes[6].set_title("Reward")
    axes[6].set_xlabel('Days')
    axes[6].set_ylabel('Reward')
    
    axes[7].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Patient trajectory plots saved successfully to {save_path}")

# ==============================================================================
# DATASET GENERATION & BENCHMARK LOOP
# ==============================================================================
class HIVDatasetGenerator:
    def __init__(self, is_pomdp=True, max_steps=100, noise_std=0.05, seed=42, test = False):
        self.is_pomdp = is_pomdp
        self.max_steps = max_steps
        self.noise_std = noise_std
        self.seed = seed
        self.test = test
        self.env = HIVSimulator(podmp=self.is_pomdp, logspace=True)
        self.env.seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def _generate_single_patient(self, patient_seed):
        # Create a fresh thread-safe simulator for each patient
       
        np.random.seed(patient_seed)
        random.seed(patient_seed)
        random_noise = np.random.uniform(-0.2, 0.01, size=4)
        random_noise = np.concatenate((random_noise, np.random.uniform(-0.01, 0.2, size=1)))
        random_noise = np.concatenate((random_noise, np.random.uniform(-0.2, 0.01, size=1)))
        random_noise = np.array([1,1,1,1,-1,1]) * np.random.uniform(0, 0.2, size=1)
        env = HIVSimulator(podmp=self.is_pomdp, logspace=True, p_init=random_noise)
        env.seed(patient_seed)
        
        
        true_state = env.reset(perturb_params=True)
        patient_obs, patient_acts = [], []

        for step in range(self.max_steps):
            noisy_obs = true_state + np.random.normal(0, self.noise_std, size=true_state.shape)
            patient_obs.append(noisy_obs)
            if step != 0:
                if self.test:
                    action = 0 if patient_seed % 2 == 0 else 3
                else:
                    action = np.random.choice([0, 1, 2, 3], p=[0.25, 0.25, 0.25, 0.25])
            else:
                action = 0
            patient_acts.append(int(action))
            true_state, reward, is_terminal, info = env.step(action)
            if is_terminal: break
        
        final_noisy_obs = true_state + np.random.normal(0, self.noise_std, size=true_state.shape)
        patient_obs.append(final_noisy_obs)

        return patient_obs, patient_acts

    def generate(self, n_patients):
        print(f"Generating {n_patients} patient trajectories in parallel...")
        results = Parallel(n_jobs=-1)(
            delayed(self._generate_single_patient)(self.seed + i)
            for i in range(n_patients)
        )
        
        all_observations = [res[0] for res in results]
        all_actions = [res[1] for res in results]

        return all_observations, all_actions


def run_hiv_benchmark_with_ci(args):
    results = {
            'EM_L1': {s: [] for s in args.train_sizes},
            'Fuzzy_L1': {s: [] for s in args.train_sizes},
            'Pyro_L1': {s: [] for s in args.train_sizes},
            'EM_LL': {s: [] for s in args.train_sizes},
            'Fuzzy_LL': {s: [] for s in args.train_sizes},
            'Pyro_LL': {s: [] for s in args.train_sizes},
            'EM_Return': {s: [] for s in args.train_sizes},
            'Fuzzy_Return': {s: [] for s in args.train_sizes},
            'EM_obs_means': {s: [] for s in args.train_sizes},
            'Fuzzy_obs_means': {s: [] for s in args.train_sizes},
            'EM_transitions': {s: [] for s in args.train_sizes},
            'Fuzzy_transitions': {s: [] for s in args.train_sizes},
        }

    for trial in range(args.n_runs):
        print(f"\n{'='*42}\n       STARTING TRIAL {trial + 1}/{args.n_runs}\n{'='*42}")
        data_gen = HIVDatasetGenerator(is_pomdp=True, noise_std=args.noise, max_steps=40, seed=42 + trial)
        train_obs, train_acts = data_gen.generate(n_patients=args.train_sizes[-1])
        data_gen.test = False
        data_gen.seed = 1000 + trial
        test_obs, test_acts = data_gen.generate(n_patients=args.n_test)

        def evaluate_configuration(n_train, train_obs, train_acts, test_obs, test_acts, trial):
            print(f"\n--- Evaluating Data Scarcity: N={n_train} ---")
            observations = train_obs[:n_train]
            actions = train_acts[:n_train]
            seed = trial*n_train
            random.seed(seed)
            np.random.seed(seed)
            
            
            em_model = POMDP_EM(n_states=args.n_states, n_actions=args.n_actions, obs_dim=args.n_obs_dim, parallel=True, seed=seed)
            hiv_var_mapping = {"T1": 0, "T2": 1, "V":  2, "E":  3} 
            
            fuzzy_model = FuzzyMAP_EM(
                n_states=args.n_states, 
                n_actions=args.n_actions, 
                obs_dim=args.n_obs_dim,  
                lambda_T=args.lambda_t, 
                lambda_O=args.lambda_o,
                fuzzy_model=HIVExpertNewModel().get_model(),
                action_mapping=hiv_action_mapping,
                hyperparameter_update_method="adaptive",
                obs_var_index=hiv_var_mapping,
                alpha_ah=0.01,
                use_fuzzy=True,
                ensure_psd=True,
                parallel=True,
                seed=seed
            )

            fuzzy_model.initialize_with_kmeans(observations, seed=seed)
            fuzzy_model.fit(observations, actions, max_iterations=args.n_iter, tolerance=1e-4)
            fuzzy_l1 = compute_avg_l1_error(fuzzy_model, test_obs, test_acts)
            fuzzy_ll = compute_log_likelihood(fuzzy_model, test_obs, test_acts)       
            print(f"Fuzzy-MAP EM (N={n_train})  -> L1: {fuzzy_l1:.3f} | Test LL: {fuzzy_ll:.3f}")

            em_model.initialize_with_kmeans(observations, seed=seed)
            em_model.fit(observations, actions, max_iterations=args.n_iter, tolerance=1e-4)
            em_l1 = compute_avg_l1_error(em_model, test_obs, test_acts)
            em_ll = compute_log_likelihood(em_model, test_obs, test_acts)
            print(f"Standard EM (N={n_train})   -> L1: {em_l1:.3f} | Test LL: {em_ll:.3f}")

            # --- RUN PLANNING EVALUATION ---
            eval_env = HIVSimulator(podmp=True, logspace=True)
            eval_env.seed(42 + trial)
            std_ret, fuzzy_ret = evaluate_planning_performance(
                true_env=eval_env,
                em_model=em_model,
                fuzzy_model=fuzzy_model,
                n_episodes=20,
                horizon=10,
                trial=trial
            )
            return n_train, em_l1, em_ll, fuzzy_l1, fuzzy_ll, std_ret, fuzzy_ret, em_model, fuzzy_model

        def evaluate_configuration_wrapper(n_train, train_obs, train_acts, test_obs, test_acts, trial):
            from joblib import parallel_backend
            # Each of the 4 configurations gets an isolated loky pool of 24 workers (4 * 24 = 96 cores)
            with parallel_backend('loky', n_jobs=24, inner_max_num_threads=1):
                return evaluate_configuration(n_train, train_obs, train_acts, test_obs, test_acts, trial)

        # Outer loop runs the 4 configurations concurrently
        config_results = Parallel(n_jobs=len(args.train_sizes), backend='loky')(
            delayed(evaluate_configuration_wrapper)(n_train, train_obs, train_acts, test_obs, test_acts, trial)
            for n_train in args.train_sizes
        )

        for n_train, em_l1, em_ll, fuzzy_l1, fuzzy_ll, std_ret, fuzzy_ret, em_model, fuzzy_model in config_results:
            results['EM_L1'][n_train].append(em_l1)
            results['Fuzzy_L1'][n_train].append(fuzzy_l1)
            results['EM_LL'][n_train].append(em_ll)
            results['Fuzzy_LL'][n_train].append(fuzzy_ll)
            results['EM_Return'][n_train].extend(std_ret)
            results['Fuzzy_Return'][n_train].extend(fuzzy_ret)
            results['EM_obs_means'][n_train].append(em_model.obs_means)
            results['Fuzzy_obs_means'][n_train].append(fuzzy_model.obs_means)
            results["EM_transitions"][n_train].append(em_model.transitions)
            results["Fuzzy_transitions"][n_train].append(fuzzy_model.transitions)
    for trial in range(args.n_runs):
        print(f"\n{'='*42}\n       MEAN and Transitions TRIAL {trial + 1}/{args.n_runs}\n{'='*42}")
        print(f"EM_obs_means: {results['EM_obs_means']}")
        print(f"Fuzzy_obs_means: {results['Fuzzy_obs_means']}")
        print(f"EM_transitions: {results['EM_transitions']}")
        print(f"Fuzzy_transitions: {results['Fuzzy_transitions']}")

    return results

def plot_results_with_ci(train_sizes, results, save_path='res/custom_hiv_benchmark_ci.png'):
    def get_stats(metric_dict):
        means = [np.mean(metric_dict[s]) for s in train_sizes]
        stds = [np.std(metric_dict[s]) for s in train_sizes]
        return np.array(means), np.array(stds)

    em_ll_mean, em_ll_std = get_stats(results['EM_LL'])
    fuz_ll_mean, fuz_ll_std = get_stats(results['Fuzzy_LL'])
    em_l1_mean, em_l1_std = get_stats(results['EM_L1'])
    fuz_l1_mean, fuz_l1_std = get_stats(results['Fuzzy_L1'])

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_sizes, em_ll_mean, label='Standard EM', marker='o')
    plt.fill_between(train_sizes, em_ll_mean - em_ll_std, em_ll_mean + em_ll_std, alpha=0.2)
    plt.plot(train_sizes, fuz_ll_mean, label='Fuzzy-MAP EM', marker='s')
    plt.fill_between(train_sizes, fuz_ll_mean - fuz_ll_std, fuz_ll_mean + fuz_ll_std, alpha=0.2)
    plt.xlabel('Training Set Size (Trajectories)')
    plt.ylabel('Held-out Test Log-Likelihood')
    plt.title('Generalization under Data Scarcity')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_sizes, em_l1_mean, label='Standard EM', marker='o')
    plt.fill_between(train_sizes, em_l1_mean - em_l1_std, em_l1_mean + em_l1_std, alpha=0.2)
    plt.plot(train_sizes, fuz_l1_mean, label='Fuzzy-MAP EM', marker='s')
    plt.fill_between(train_sizes, fuz_l1_mean - fuz_l1_std, fuz_l1_mean + fuz_l1_std, alpha=0.2)
    plt.xlabel('Training Set Size (Trajectories)')
    plt.ylabel('One-Step-Ahead L1 Error')
    plt.title('Predictive Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\nStatistical plots saved successfully to {save_path}")

def plot_returns_bar_chart(train_sizes, results, save_path='res/cumulative_returns_bar.png'):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    
    em_returns_dict = results.get('EM_Return', {})
    fuzzy_returns_dict = results.get('Fuzzy_Return', {})
    
    if not em_returns_dict or not fuzzy_returns_dict:
        return
        
    # Prepare data for Seaborn DataFrame
    data = []
    for s in train_sizes:
        for val in em_returns_dict.get(s, []):
            data.append({'Training Size': str(s), 'Average Cumulative Return': val, 'Model': 'Standard EM'})
        for val in fuzzy_returns_dict.get(s, []):
            data.append({'Training Size': str(s), 'Average Cumulative Return': val, 'Model': 'Fuzzy-MAP EM'})
            
    if not data:
        return
        
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(10, 6))
    
    # Use seaborn barplot which automatically handles the confidence intervals (95% CI by default)
    sns.barplot(
        data=df,
        x='Training Size',
        y='Average Cumulative Return',
        hue='Model',
        capsize=.08,
        palette={'Standard EM': 'blue', 'Fuzzy-MAP EM': 'red'},
        alpha=0.7
    )
    
    plt.title('Cumulative Return per Scenario with 95% Confidence')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Returns bar chart saved to {save_path}")


def run_grid_search(args):
    import itertools
    import pandas as pd
    import seaborn as sns
    import numpy as np
    import matplotlib.pyplot as plt
    from joblib import parallel_backend, Parallel, delayed
    
    lambda_vals = [0, 0.1, 5, 10]
    n_patients = 15
    n_trials = 3
    
    def evaluate_grid_point(lt, lo, trial, train_obs, train_acts, test_obs, test_acts):
        hiv_var_mapping = {"T1": 0, "T2": 1, "V":  2, "E":  3} 
        fuzzy_model = FuzzyMAP_EM(
            n_states=args.n_states, 
            n_actions=args.n_actions, 
            obs_dim=args.n_obs_dim,  
            lambda_T=lt, 
            lambda_O=lo,
            fuzzy_model=HIVExpertNewModel().get_model(),
            action_mapping=hiv_action_mapping,
            hyperparameter_update_method="adaptive",
            obs_var_index=hiv_var_mapping,
            alpha_ah=0.05,
            eb_learning_rate_O=lo,
            eb_learning_rate_T=lt,
            use_fuzzy=True,
            ensure_psd=True,
            parallel=True,
            seed=42
        )
        
        fuzzy_model.fit(train_obs, train_acts, max_iterations=args.n_iter, tolerance=1e-4)
        fuzzy_l1 = compute_avg_l1_error(fuzzy_model, test_obs, test_acts)
        fuzzy_ll = compute_log_likelihood(fuzzy_model, test_obs, test_acts)
        
        discount_factor = 0.99
        fuzzy_params = {
            "T": fuzzy_model.transitions,
            "mu": fuzzy_model.obs_means,
            "Sigma": fuzzy_model.obs_covs
        }
        fuzzy_env = LearnedHIVEnvironment("fuzzy_env", fuzzy_params, my_hiv_reward_fn, discount_factor=discount_factor)
        fuzzy_sampler = DiscreteActionSampler(fuzzy_env.get_actions())
        
        planner_config = {
            "n_simulations":1000,
            "depth": 3,
            "discount_factor": 0.95,
            "exploration_constant": 10000,
            "k_o": 4.868167504611893,
            "k_a": 4.203111794128549,
            "alpha_o": 0.3982346933640448,
            "alpha_a": 0.46669636595271263,
        }
        
        fuzzy_policy = POMCPOW(fuzzy_env, action_sampler=fuzzy_sampler, name="POMCP_Fuzzy", **planner_config)
        fuzzy_initial_belief = get_initial_belief(fuzzy_env, n_particles=50)  # Reduced particles
        
        unhealthy_steady_state = [163573., 5., 11945., 46., 63919., 24.]
        eval_env = TrueHIVEnvironment(initial_biological_state=unhealthy_steady_state, discount_factor=discount_factor)
        
        episode_returns = []
        for ep in range(5):  # Run 5 episodes to estimate policy reward
            ret, _, _ = evaluate_cross_environment(eval_env, fuzzy_policy, fuzzy_initial_belief, episode=ep, trial = trial)
            episode_returns.append(ret)
            
        mean_reward = np.mean(episode_returns)
        
        print(f"[Trial {trial}] lambda_T={lt}, lambda_O={lo} -> L1={fuzzy_l1:.3f}, LL={fuzzy_ll:.3f}, Reward={mean_reward:.2f}")
        return {'lambda_t': lt, 'lambda_o': lo, 'trial': trial, 'L1': fuzzy_l1, 'LL': fuzzy_ll, 'Reward': mean_reward}

    def evaluate_wrapper(lt, lo, trial, train_obs, train_acts, test_obs, test_acts):
        with parallel_backend('loky', n_jobs=30, inner_max_num_threads=1):
            return evaluate_grid_point(lt, lo, trial, train_obs, train_acts, test_obs, test_acts)

    trial_data = {}
    print(f"Generating data for {n_trials} trials...")
    with parallel_backend('loky', n_jobs=96, inner_max_num_threads=1):
        for trial in range(n_trials):
            data_gen = HIVDatasetGenerator(is_pomdp=True, noise_std=args.noise, max_steps=50, seed=42 + trial)
            train_obs, train_acts = data_gen.generate(n_patients=n_patients)
            data_gen.test = True
            test_obs, test_acts = data_gen.generate(n_patients=args.n_test)
            trial_data[trial] = (train_obs, train_acts, test_obs, test_acts)

    grid = list(itertools.product(lambda_vals, lambda_vals, range(n_trials)))
    
    print(f"\n{'='*40}\n   STARTING GRID SEARCH ({len(grid)} tasks)\n{'='*40}")
    out = Parallel(n_jobs=3, backend='loky')(
        delayed(evaluate_wrapper)(
            lt, lo, trial, 
            trial_data[trial][0], trial_data[trial][1], 
            trial_data[trial][2], trial_data[trial][3]
        ) for lt, lo, trial in grid
    )
    
    df = pd.DataFrame(out)

    mean_df = df.groupby(['lambda_t', 'lambda_o'])[['L1', 'LL', 'Reward']].mean().reset_index()
    
    # Identify the best hyperparameter configuration by average reward
    best_idx = mean_df['Reward'].idxmax()
    best_config = mean_df.loc[best_idx]
    print(f"\n{'='*40}\n   BEST CONFIGURATION BY REWARD:\n"
          f"   lambda_T: {best_config['lambda_t']}\n"
          f"   lambda_O: {best_config['lambda_o']}\n"
          f"   Best Mean Reward: {best_config['Reward']:.2f}\n"
          f"   Corresponding L1: {best_config['L1']:.3f}\n"
          f"   {'='*40}\n")
    pivot_l1 = mean_df.pivot(index='lambda_t', columns='lambda_o', values='L1')
    pivot_ll = mean_df.pivot(index='lambda_t', columns='lambda_o', values='LL')
    pivot_reward = mean_df.pivot(index='lambda_t', columns='lambda_o', values='Reward')
    
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    sns.heatmap(pivot_l1, annot=True, cmap='viridis_r', ax=axes[0], fmt=".3f")
    axes[0].set_title('Grid Search: Average L1 Error')
    axes[0].set_xlabel('Lambda_O')
    axes[0].set_ylabel('Lambda_T')
    
    sns.heatmap(pivot_ll, annot=True, cmap='viridis', ax=axes[1], fmt=".2f")
    axes[1].set_title('Grid Search: Average Log-Likelihood')
    axes[1].set_xlabel('Lambda_O')
    axes[1].set_ylabel('Lambda_T')
    
    sns.heatmap(pivot_reward, annot=True, cmap='viridis', ax=axes[2], fmt=".2f")
    axes[2].set_title('Grid Search: Average Reward')
    axes[2].set_xlabel('Lambda_O')
    axes[2].set_ylabel('Lambda_T')

    plt.tight_layout()
    plt.savefig('res/grid_search_heatmaps.png', dpi=300)
    print("Grid search complete. Heatmaps saved to res/grid_search_heatmaps.png")
    
    df.to_csv('res/grid_search_results.csv', index=False)
    return df


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Run Statistical HIV Benchmark for Fuzzy-MAP EM")
    parser.add_argument("--n_runs", type=int, default=15, help="Number of independent trials to compute confidence intervals")
    parser.add_argument("--run_planning", action="store_true", help="Run POMDPPlanners evaluation to compare accumulated returns")
    parser.add_argument("--n_states", type=int, default=5, help="Discrete latent phases")
    parser.add_argument("--n_actions", type=int, default=4, help="0: None, 1: RTI, 2: PI, 3: Both")
    parser.add_argument("--n_obs_dim", type=int, default=4, help="Masked observations (T1, T2, Viral Load, E)")
    parser.add_argument("--n_iter", type=int, default=250, help="Maximum EM iterations")
    parser.add_argument("--lambda_t", type=float, default=6, help="Transition fuzzy weight")
    parser.add_argument("--lambda_o", type=float, default=0.1, help="Observation fuzzy weight")
    parser.add_argument("--noise", type=float, default=0.1, help="Gaussian noise added to standardized observations")
    parser.add_argument("--train_sizes", type=int, nargs='+', default=[20, 25, 30])
    parser.add_argument("--n_test", type=int, default=200)
    parser.add_argument("--grid_search", action="store_true", help="Run hyperparameter grid search for lambda_T and lambda_O")

    args = parser.parse_args()
    
    if args.grid_search:
        run_grid_search(args)
    else:
        results = run_hiv_benchmark_with_ci(args)
        plot_results_with_ci(args.train_sizes, results)
        plot_returns_bar_chart(args.train_sizes, results)
        
