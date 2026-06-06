import argparse
import numpy as np
import matplotlib.pyplot as plt
from fuzzy.hiv_fuzzy import HIVExpert5DModel

# Import the custom simulator instead of whynot
from hiv_simulator import HIVSimulator

# Import your existing models and metrics
from models.trainable.pomdp_EM import PomdpEM as POMDP_EM
from models.trainable.fuzzy_EM import FuzzyPOMDP as FuzzyMAP_EM
from utils.metrics import compute_avg_l1_error, compute_log_likelihood, plot_state_observation_distributions


class HIVDatasetGenerator:
    """
    A streamlined data generator for creating offline HIV treatment trajectories.
    Automatically handles POMDP masking, clinical noise, and formatting for EM models.
    """
    def __init__(self, is_pomdp=True, max_steps=100, noise_std=0.05, seed=42):
        self.is_pomdp = is_pomdp
        self.max_steps = max_steps
        self.noise_std = noise_std
        
        # Initialize simulator (podmp=True applies the T1, T2, V, E mask automatically)
        self.env = HIVSimulator(podmp=self.is_pomdp, logspace=True)
        
        # Set seeds for reproducible datasets
        self.env.seed(seed)
        np.random.seed(seed)

    def generate(self, n_patients):
        """
        Generates clean observation and action sequences for a set number of patients.
        
        Returns:
            observations: List of lists containing patient observation arrays.
            actions: List of lists containing patient action integers.
        """
        all_observations = []
        all_actions = []

        for _ in range(n_patients):
            # perturb_params=True ensures biological diversity between patients
            true_state = self.env.reset(perturb_params=True)
            
            patient_obs = []
            patient_acts = []

            for step in range(self.max_steps):
                # 1. Add measurement noise to the observation
                noisy_obs = true_state + np.random.normal(0, self.noise_std, size=true_state.shape)
                patient_obs.append(noisy_obs)

                # 2. Sample an action (Offline RL random behavioral policy)
                action = np.random.randint(0, self.env.num_actions)
                patient_acts.append(int(action))

                # 3. Step the environment
                true_state, reward, is_terminal, info = self.env.step(action)

                if is_terminal:
                    break
            
            # Record the final observation after the last action
            final_noisy_obs = true_state + np.random.normal(0, self.noise_std, size=true_state.shape)
            patient_obs.append(final_noisy_obs)

            all_observations.append(patient_obs)
            all_actions.append(patient_acts)

        return all_observations, all_actions

def run_hiv_benchmark_with_ci(args):
    """
    Executes the data scarcity benchmark across multiple independent trials.
    """
    # Dictionary to store lists of results for each training size
    results = {
        'EM_L1': {s: [] for s in args.train_sizes},
        'Fuzzy_L1': {s: [] for s in args.train_sizes},
        'EM_LL': {s: [] for s in args.train_sizes},
        'Fuzzy_LL': {s: [] for s in args.train_sizes}
    }


    for trial in range(args.n_runs):
        print(f"\n==========================================")
        print(f"       STARTING TRIAL {trial + 1}/{args.n_runs}")
        print(f"==========================================")
        data_gen = HIVDatasetGenerator(is_pomdp=True, noise_std=0.05, max_steps=5, seed=42 + trial)
        train_obs, train_acts = data_gen.generate(n_patients=50)
        test_obs, test_acts = data_gen.generate(n_patients=args.n_test)

        #data_gen_pomdp = HIVDatasetGenerator(is_pomdp=False, noise_std=0.001, max_steps=10)
        #real_pomdp_data_train_obs, real_pomdp_data_train_acts = data_gen_pomdp.generate(n_patients=1000)
        #test_pomdp_data_train_obs, test_pomdp_data_train_acts = data_gen_pomdp.generate(n_patients=1000)
        #real_pomdp = POMDP_EM(n_states=args.n_states, n_actions=args.n_actions, obs_dim=6, verbose=True)
        #real_pomdp.initialize_with_kmeans(real_pomdp_data_train_obs)
        #print("\nFitting Real POMDP with Full Data...")       
        #real_pomdp.fit(real_pomdp_data_train_obs, real_pomdp_data_train_acts, max_iterations=100)
        #real_pomdp_l1 = compute_avg_l1_error(real_pomdp, test_pomdp_data_train_obs, test_pomdp_data_train_acts)
        #real_pomdp_ll = compute_log_likelihood(real_pomdp, test_pomdp_data_train_obs, test_pomdp_data_train_acts)
        #print(f"Real POMDP    -> L1: {real_pomdp_l1:.3f} | Test LL: {real_pomdp_ll:.3f}")

        for n_train in args.train_sizes:
            print(f"\n--- Evaluating Data Scarcity: N={n_train} ---")
            observations = train_obs[:n_train]
            actions = train_acts[:n_train]
            
            # Initialize Models dynamically
            em_model = POMDP_EM(n_states=args.n_states, n_actions=args.n_actions, obs_dim=args.n_obs_dim)

            hiv_var_mapping = {"T1": 0, "T2": 1, "V":  2, "E":  3} 
            fuzzy_model = FuzzyMAP_EM(
                n_states=args.n_states, 
                n_actions=args.n_actions, 
                obs_dim=4,  
                lambda_T=args.lambda_t, 
                lambda_O=args.lambda_o,
                fuzzy_model=HIVExpert5DModel().get_model(), # Use the new 5D class
                hyperparameter_update_method="adaptive",
                obs_var_index=hiv_var_mapping ,
                use_fuzzy=True,
                ensure_psd=True,
                parallel=False,
            )

            #em_model.initialize_with_kmeans(observations)           
            em_model.fit(observations, actions, max_iterations=args.n_iter, tolerance=1e-4)
            em_l1 = compute_avg_l1_error(em_model, test_obs, test_acts)
            em_ll = compute_log_likelihood(em_model, test_obs, test_acts)

            print(f"Standard EM   -> L1: {em_l1:.3f} | Test LL: {em_ll:.3f}")

            #fuzzy_model.initialize_with_kmeans(observations) 
            fuzzy_model.fit(observations, actions, max_iterations=args.n_iter, tolerance=1e-4)
            fuzzy_l1 = compute_avg_l1_error(fuzzy_model, test_obs, test_acts)
            fuzzy_ll = compute_log_likelihood(fuzzy_model, test_obs, test_acts)       
            
            results['EM_L1'][n_train].append(em_l1)
            results['Fuzzy_L1'][n_train].append(fuzzy_l1)
            results['EM_LL'][n_train].append(em_ll)
            results['Fuzzy_LL'][n_train].append(fuzzy_ll)

            plot_state_observation_distributions(em_model, ["T1", "T2", "V", "E"], save_path=f'res/em_trial{trial+1}_n{n_train}.png')
            plot_state_observation_distributions(fuzzy_model, ["T1", "T2", "V", "E"], save_path=f'res/fuzzy_trial{trial+1}_n{n_train}.png')
            
            print(f"Fuzzy-MAP EM  -> L1: {fuzzy_l1:.3f} | Test LL: {fuzzy_ll:.3f}")

    return results

def plot_results_with_ci(train_sizes, results, save_path='res/custom_hiv_benchmark_ci.png'):
    
    # Helper function to extract mean and standard deviation
    def get_stats(metric_dict):
        means = [np.mean(metric_dict[s]) for s in train_sizes]
        stds = [np.std(metric_dict[s]) for s in train_sizes]
        return np.array(means), np.array(stds)

    em_ll_mean, em_ll_std = get_stats(results['EM_LL'])
    fuz_ll_mean, fuz_ll_std = get_stats(results['Fuzzy_LL'])
    
    em_l1_mean, em_l1_std = get_stats(results['EM_L1'])
    fuz_l1_mean, fuz_l1_std = get_stats(results['Fuzzy_L1'])

    plt.figure(figsize=(12, 5))

    # --- Log-Likelihood Plot ---
    plt.subplot(1, 2, 1)
    plt.plot(train_sizes, em_ll_mean, label='Standard EM', marker='o', color='tab:blue')
    plt.fill_between(train_sizes, em_ll_mean - em_ll_std, em_ll_mean + em_ll_std, color='tab:blue', alpha=0.2)
    
    plt.plot(train_sizes, fuz_ll_mean, label='Fuzzy-MAP EM', marker='s', color='tab:orange')
    plt.fill_between(train_sizes, fuz_ll_mean - fuz_ll_std, fuz_ll_mean + fuz_ll_std, color='tab:orange', alpha=0.2)
    
    plt.xlabel('Training Set Size (Trajectories)')
    plt.ylabel('Held-out Test Log-Likelihood')
    plt.title('Generalization under Data Scarcity')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # --- L1 Error Plot ---
    plt.subplot(1, 2, 2)
    plt.plot(train_sizes, em_l1_mean, label='Standard EM', marker='o', color='tab:blue')
    plt.fill_between(train_sizes, em_l1_mean - em_l1_std, em_l1_mean + em_l1_std, color='tab:blue', alpha=0.2)
    
    plt.plot(train_sizes, fuz_l1_mean, label='Fuzzy-MAP EM', marker='s', color='tab:orange')
    plt.fill_between(train_sizes, fuz_l1_mean - fuz_l1_std, fuz_l1_mean + fuz_l1_std, color='tab:orange', alpha=0.2)
    
    plt.xlabel('Training Set Size (Trajectories)')
    plt.ylabel('One-Step-Ahead L1 Error')
    plt.title('Predictive Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\nStatistical plots saved successfully to {save_path}")

    #plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Statistical HIV Benchmark for Fuzzy-MAP EM")
    
    # Run configuration
    parser.add_argument("--n_runs", type=int, default=5, help="Number of independent trials to compute confidence intervals")
    
    # POMDP Dimensions
    parser.add_argument("--n_states", type=int, default=3, help="Discrete latent phases")
    parser.add_argument("--n_actions", type=int, default=4, help="0: None, 1: RTI, 2: PI, 3: Both")
    
    # Updated default to 4 dimensions based on the HIVSimulator mask [T1, T2, V, E]
    parser.add_argument("--n_obs_dim", type=int, default=4, help="Masked observations (T1, T2, Viral Load, E)")
    
    # Hyperparameters
    parser.add_argument("--n_iter", type=int, default=500, help="Maximum EM iterations")
    parser.add_argument("--lambda_t", type=float, default=10, help="Transition fuzzy weight")
    parser.add_argument("--lambda_o", type=float, default=0.2, help="Observation fuzzy weight")
    parser.add_argument("--noise", type=float, default=0.001, help="Gaussian noise added to standardized observations")
    
    # Dataset splits
    parser.add_argument("--train_sizes", type=int, nargs='+', default=[5, 10, 25])
    parser.add_argument("--n_test", type=int, default=800)

    args = parser.parse_args()

    # Run Benchmark & Plot
    results = run_hiv_benchmark_with_ci(args)
    plot_results_with_ci(args.train_sizes, results)