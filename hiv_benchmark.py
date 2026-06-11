import argparse
import torch
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import random
from fuzzy.hiv_fuzzy import HIVExpert5DModel

# Import the custom simulator instead of whynot
from hiv_simulator import HIVSimulator
from utils.utils import my_hiv_reward_fn

# Import your existing models and metrics
from models.trainable.pomdp_EM import PomdpEM as POMDP_EM
from models.trainable.fuzzy_EM import FuzzyPOMDP as FuzzyMAP_EM
from models.trainable.vb_pomdp import VariationalBayesianPOMDP as VB_POMDP
from utils.metrics import compute_avg_l1_error, compute_log_likelihood
from pathlib import Path

# ==============================================================================
# yaacovpariente/POMDPPlanners INTEGRATION
# ==============================================================================

from POMDPPlanners.planners.mcts_planners.pft_dpw import PFT_DPW
from POMDPPlanners.planners.mcts_planners.pomcp import POMCP
from POMDPPlanners.planners.mcts_planners.pomcpow import POMCPOW
from POMDPPlanners.simulations.episodes import run_episode
from POMDPPlanners.utils.action_samplers import DiscreteActionSampler
from POMDPPlanners.core.environment import (
    Environment,
    DiscreteActionsEnvironment,
    SpaceInfo,
    SpaceType,
)
from POMDPPlanners.core.belief import get_initial_belief
from POMDPPlanners.utils.belief_factory import create_environment_belief
from POMDPPlanners.simulations.simulation_apis.local_simulations_api import LocalSimulationsAPI
from POMDPPlanners.core.simulation import EnvironmentRunParams
from POMDPPlanners.utils.logger import get_logger


POMDPPLANNERS_AVAILABLE = True
#except ImportError:
#    POMDPPLANNERS_AVAILABLE = False
#    print("Warning: POMDPPlanners not found. Planning benchmark will be skipped unless installed.")
from envs.custom_env_to_plan import LearnedHIVEnvironment
from envs.true_hiv_pomdp import TrueHIVEnvironment

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

def evaluate_cross_environment(true_env, policy, initial_belief, episode = 1, max_steps=100):
    """
    Evaluates a policy trained on a Learned Environment inside a True Environment.
    """
    # 1. The Body: Initialize true biological reality (6D array)
    seed =42 * episode
    np.random.seed(seed)
    import random
    random.seed(seed)
    true_state = true_env.initial_state_dist().sample()[0]
    
    # 2. The Brain: Initialize the AI's mental state (over states 0, 1, 2)
    # We use .copy() so parallel episodes don't share the same memory
    import copy
    belief = copy.deepcopy(initial_belief)
    
    total_reward = 0.0
    
    states_history = [true_state]
    rewards_history = []
    
    for step in range(max_steps):
        try:
            action = policy.plan(belief)
        except AttributeError:
            action = policy.action(belief)
        
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

def evaluate_planning_performance(true_env, em_model, fuzzy_model, n_episodes=50, horizon=15):
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
    

    std_env = LearnedHIVEnvironment("std_env", std_params, my_hiv_reward_fn)
    fuzzy_env = LearnedHIVEnvironment("fuzzy_env", fuzzy_params, my_hiv_reward_fn)

    std_sampler = DiscreteActionSampler(std_env.get_actions())
    fuzzy_sampler = DiscreteActionSampler(fuzzy_env.get_actions())

    unhealthy_steady_state = [163573., 5., 11945., 46., 63919., 24.]
    true_env = TrueHIVEnvironment(initial_biological_state=unhealthy_steady_state)

    # 2. Configure the planners
    # Note: You need to tune these hyperparameters based on your HIV benchmark
    planner_config = {
        "n_simulations":100,
        "depth": 50,
        "discount_factor": 0.95,
        "exploration_constant": 1.0,
        "k_o": 10,
        "k_a": 4,
        "alpha_o": 0.01,
        "alpha_a": 0.01,
    }

    # Swap POMCP for PFT_DPW
    std_policy = POMCPOW(std_env,action_sampler=std_sampler, name="PFT_DPW_Standard", **planner_config)
    fuzzy_policy = POMCPOW(fuzzy_env,action_sampler=fuzzy_sampler, name="PFT_DPW_Fuzzy", **planner_config)

    # 3. Setup Initial Beliefs (e.g., Uniform particle belief)
    # You can customize this based on the POMDPPlanners documentation
    initial_belief = get_initial_belief(fuzzy_env, n_particles=100)

    # 4. Run the Evaluation
    api = LocalSimulationsAPI()
    print("Evaluating Standard EM Model...")
    #std_results = api.run_multiple_environments_and_policies(
    #    environment_run_params=[
    #        EnvironmentRunParams(
    #            environment=std_env,
    #            belief=initial_belief,
    #            policies=[std_policy],
    #            num_episodes=100,
    #            num_steps=100
    #        )
    #    ],
    #    alpha=0.1,  # Required by the API for risk metrics (CVaR/VaR)
    #    confidence_interval_level=0.95,
    #    experiment_name="Standard_EM_Evaluation"
    #)
    logger = get_logger("basic_example",
                    output_dir=Path("/tmp/test_logs"),
                    console_output=True)

    #history = run_episode(
    #    environment=true_env,
    #    policy=fuzzy_policy,
    #    initial_belief=initial_belief,
    #    num_steps=5,
    #    logger=logger
    #)

    environment_run_params=[
        EnvironmentRunParams(
            environment=true_env,
            belief=initial_belief,
            policies=[std_policy, fuzzy_policy],
            num_episodes=15,
            num_steps=10
        ),
    ]

    print("Evaluating Fuzzy-MAP EM Model...")
    print("Evaluating Standard EM Model in REALITY...")
    std_results = Parallel(n_jobs=-1)(
        delayed(evaluate_cross_environment)(true_env, std_policy, initial_belief, episode = ep) 
        for ep in range(100) # 100 episodes
    )
    fuzzy_results = Parallel(n_jobs=-1)(
        delayed(evaluate_cross_environment)(true_env, fuzzy_policy, initial_belief, episode = ep) 
        for ep in range(100) # 100 episodes
    )
    
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
        
        axes[i].plot(std_mean, label='Standard EM', color='blue')
        axes[i].fill_between(range(len(std_mean)), std_mean - std_std, std_mean + std_std, alpha=0.2, color='blue')
        
        axes[i].plot(fuz_mean, label='Fuzzy-MAP EM', color='orange')
        axes[i].fill_between(range(len(fuz_mean)), fuz_mean - fuz_std, fuz_mean + fuz_std, alpha=0.2, color='orange')
        
        axes[i].set_title(state_labels[i])
        axes[i].set_xlabel('Time Steps')
        axes[i].set_ylabel('Log10(Cell Count / Virus Load)')
        if i == 0:
            axes[i].legend()
            
    with np.errstate(all='ignore'):
        std_r_mean = np.nanmean(std_rewards_padded, axis=0)
        std_r_std = np.nanstd(std_rewards_padded, axis=0)
        fuz_r_mean = np.nanmean(fuzzy_rewards_padded, axis=0)
        fuz_r_std = np.nanstd(fuzzy_rewards_padded, axis=0)
    
    axes[6].plot(std_r_mean, label='Standard EM', color='blue')
    axes[6].fill_between(range(len(std_r_mean)), std_r_mean - std_r_std, std_r_mean + std_r_std, alpha=0.2, color='blue')
    
    axes[6].plot(fuz_r_mean, label='Fuzzy-MAP EM', color='orange')
    axes[6].fill_between(range(len(fuz_r_mean)), fuz_r_mean - fuz_r_std, fuz_r_mean + fuz_r_std, alpha=0.2, color='orange')
    
    axes[6].set_title("Reward")
    axes[6].set_xlabel('Time Steps')
    axes[6].set_ylabel('Reward')
    
    axes[7].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Patient trajectory plots saved successfully to {save_path}")

# ==============================================================================
# DATASET GENERATION & BENCHMARK LOOP
# ==============================================================================
class HIVDatasetGenerator:
    def __init__(self, is_pomdp=True, max_steps=100, noise_std=0.05, seed=42):
        self.is_pomdp = is_pomdp
        self.max_steps = max_steps
        self.noise_std = noise_std
        self.seed = seed
        self.env = HIVSimulator(podmp=self.is_pomdp, logspace=True)
        self.env.seed(seed)
        np.random.seed(seed)
        import random
        random.seed(seed)

    def _generate_single_patient(self, patient_seed):
        # Create a fresh thread-safe simulator for each patient
        env = HIVSimulator(podmp=self.is_pomdp, logspace=True)
        env.seed(patient_seed)
        np.random.seed(patient_seed)
        import random
        random.seed(patient_seed)
        
        true_state = env.reset(perturb_params=True)
        patient_obs, patient_acts = [], []

        for step in range(self.max_steps):
            noisy_obs = true_state + np.random.normal(0, self.noise_std, size=true_state.shape)
            patient_obs.append(noisy_obs)
            action = np.random.randint(0, env.num_actions)
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
        'EM_LL': {s: [] for s in args.train_sizes},
        'Fuzzy_LL': {s: [] for s in args.train_sizes}
    }

    for trial in range(args.n_runs):
        print(f"\n{'='*42}\n       STARTING TRIAL {trial + 1}/{args.n_runs}\n{'='*42}")
        data_gen = HIVDatasetGenerator(is_pomdp=True, noise_std=args.noise, max_steps=100, seed=42 + trial)
        train_obs, train_acts = data_gen.generate(n_patients=300)
        test_obs, test_acts = data_gen.generate(n_patients=args.n_test)

        for n_train in args.train_sizes:
            print(f"\n--- Evaluating Data Scarcity: N={n_train} ---")
            observations = train_obs[:n_train]
            actions = train_acts[:n_train]
            
            em_model = POMDP_EM(n_states=args.n_states, n_actions=args.n_actions, obs_dim=args.n_obs_dim, parallel=True)
            hiv_var_mapping = {"T1": 0, "T2": 1, "V":  2, "E":  3} 
            
            fuzzy_model = FuzzyMAP_EM(
                n_states=args.n_states, 
                n_actions=args.n_actions, 
                obs_dim=args.n_obs_dim,  
                lambda_T=args.lambda_t, 
                lambda_O=args.lambda_o,
                fuzzy_model=HIVExpert5DModel().get_model(),
                hyperparameter_update_method="adaptive",
                obs_var_index=hiv_var_mapping,
                alpha_ah=0.1,
                use_fuzzy=True,
                ensure_psd=True,
                parallel=False,
            )

            em_model.fit(observations, actions, max_iterations=args.n_iter, tolerance=1e-4)
            em_l1 = compute_avg_l1_error(em_model, test_obs, test_acts)
            em_ll = compute_log_likelihood(em_model, test_obs, test_acts)
            print(f"Standard EM   -> L1: {em_l1:.3f} | Test LL: {em_ll:.3f}")

            fuzzy_model.fit(observations, actions, max_iterations=args.n_iter, tolerance=1e-4)
            fuzzy_l1 = compute_avg_l1_error(fuzzy_model, test_obs, test_acts)
            fuzzy_ll = compute_log_likelihood(fuzzy_model, test_obs, test_acts)       
            print(f"Fuzzy-MAP EM  -> L1: {fuzzy_l1:.3f} | Test LL: {fuzzy_ll:.3f}")

            results['EM_L1'][n_train].append(em_l1)
            results['Fuzzy_L1'][n_train].append(fuzzy_l1)
            results['EM_LL'][n_train].append(em_ll)
            results['Fuzzy_LL'][n_train].append(fuzzy_ll)
            
            # --- RUN PLANNING EVALUATION VIA LOCAL_SIMULATION_API ---
            if True:
                eval_env = HIVSimulator(podmp=True, logspace=True)
                eval_env.seed(88 + trial)
                evaluate_planning_performance(
                    true_env=eval_env,
                    em_model=em_model,
                    fuzzy_model=fuzzy_model,
                    n_episodes=20,
                    horizon=10
                )

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Statistical HIV Benchmark for Fuzzy-MAP EM")
    parser.add_argument("--n_runs", type=int, default=1, help="Number of independent trials to compute confidence intervals")
    parser.add_argument("--run_planning", action="store_true", help="Run POMDPPlanners evaluation to compare accumulated returns")
    parser.add_argument("--n_states", type=int, default=2, help="Discrete latent phases")
    parser.add_argument("--n_actions", type=int, default=4, help="0: None, 1: RTI, 2: PI, 3: Both")
    parser.add_argument("--n_obs_dim", type=int, default=4, help="Masked observations (T1, T2, Viral Load, E)")
    parser.add_argument("--n_iter", type=int, default=500, help="Maximum EM iterations")
    parser.add_argument("--lambda_t", type=float, default=10, help="Transition fuzzy weight")
    parser.add_argument("--lambda_o", type=float, default=0.25, help="Observation fuzzy weight")
    parser.add_argument("--noise", type=float, default=0.1, help="Gaussian noise added to standardized observations")
    parser.add_argument("--train_sizes", type=int, nargs='+', default=[300])
    parser.add_argument("--n_test", type=int, default=800)

    args = parser.parse_args()
    results = run_hiv_benchmark_with_ci(args)
    plot_results_with_ci(args.train_sizes, results)