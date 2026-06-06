import re

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm


def _from_beta_to_multivariate_normal(obs_model_real, state_name):
    """
    Convert beta parameters to a multivariate normal distribution.
    :param obs_model_real: The observation model containing beta parameters.
    :param state_name: The name of the state for which to convert the parameters.
    :return: A GaussianMixture object representing the multivariate normal distribution.
    """

    means = []
    variances = []
    for dim, p in obs_model_real[0][state_name].items():
        a, b = p
        # Beta mean and variance
        means.append(a / (a + b))
        variances.append((a * b) / ((a + b) ** 2 * (a + b + 1)))

    cov_MVN = np.diag(variances)
    real_MVN = multivariate_normal(mean=means, cov=cov_MVN)

    return real_MVN


def _compute_KL_divergence(mvn_p, mvn_q):
    """
    Compute the KL divergence between two Gaussian Mixture Models.
    :param mvn_p: The first Gaussian Mixture Model (GMM).
    :param mvn_q: The second Gaussian Mixture Model (GMM).
    :return: Closed-form KL divergence D_KL(P \| Q) for single-component Gaussian GMMs.
    """
    k = mvn_p.mean.shape[0]

    cov0 = mvn_p.cov
    cov1 = mvn_q.cov
    cov0 += 1e-6 * np.eye(cov0.shape[0])
    cov1 += 1e-6 * np.eye(cov1.shape[0])

    cov1_inv = np.linalg.inv(cov1)

    trace_term = np.trace(cov1_inv @ cov0)
    diff = mvn_q.mean - mvn_p.mean
    term = diff @ cov1_inv @ diff.T

    (sign0, logdet0) = np.linalg.slogdet(cov0)
    (sign1, logdet1) = np.linalg.slogdet(cov1)
    log_det_term = logdet1 - logdet0
    kl_approx = 0.5 * (trace_term + term + log_det_term - k)
    return kl_approx


def _mvn_definition_from_POMDP(mvns, state):
    """
    Format the MVN using the scipy stats lib.
    :param mvns: A set of MVN representing the multivariate normal distribution in pomdp.
    :param state: The  state to convert.
    :return: A MVN object.
    """
    means = mvns.obs_means[state]
    covariances = mvns.obs_covs[state]

    mvn = multivariate_normal(mean=means, cov=covariances)
    return mvn


def _from_mean_cov_to_gmm(means, cov):
    """
    Convert a multivariate normal distribution to a Gaussian Mixture Model.
    :param mvns: A set of MVN representing the multivariate normal distribution in pomdp.
    :param state: The  state to convert.
    :return: A GaussianMixture object.
    """

    return multivariate_normal(mean=means, cov=cov)


def _from_dist_to_mvn(params, dist_type, state_name=None):
    """
    Convert distribution parameters to a multivariate normal distribution.
    :param params: The distribution parameters.
    :param dist_type: Type of distribution ("beta" or "mvn").
    :return: A multivariate normal distribution.
    """
    if dist_type == "mvn":
        mean = params[0][state_name]["mean"]
        cov = params[0][state_name]["cov"]
        return _from_mean_cov_to_gmm(mean, cov)
    elif dist_type == "beta":
        return _from_beta_to_multivariate_normal(params, state_name)
    else:
        raise ValueError(f"Unsupported distribution type: {dist_type}")


def normalize_cost_matrix(cost_matrix):
    if cost_matrix.max() == cost_matrix.min(): return np.zeros_like(cost_matrix)
    return (cost_matrix - cost_matrix.min()) / (cost_matrix.max() - cost_matrix.min())


def match_state_hungarian(learned_model, true_transitions, true_observations, states, dist_type="beta", alpha=0.35):
    """
    Match the states of the learned model to the true model using the Hungarian algorithm.
    :param learned_model: The learned model with transition and observation parameters
    :param true_transitions: The true transition parameters.
    :param true_observations: The true observation parameters.
    :param alpha: Weighting factor between transition and observation KL divergences.
    :param dist_type: Type of distribution for observations ("beta" or "mvn").
    :param states: List of state names.
    :return: A list of matched state indices.
    """
    n_states = learned_model.n_states

    cost_trans = np.zeros((n_states, n_states))
    cost_obs = np.zeros((n_states, n_states))
    for row in range(n_states):
        for col in range(n_states):
            learned_obs_mvn = _mvn_definition_from_POMDP(learned_model, state=row)
            true_obs_mvn = _from_dist_to_mvn(true_observations, dist_type, state_name=states[col])

            KL_row_col = _compute_KL_divergence(learned_obs_mvn, true_obs_mvn)
            KL_col_row = _compute_KL_divergence(true_obs_mvn, learned_obs_mvn)
            cost_obs[row, col] = 0.5 * (KL_row_col + KL_col_row)

            cost_trans[row, col] = np.sum(np.abs(learned_model.transitions[row] - true_transitions[col]))

    cost_trans = normalize_cost_matrix(cost_trans)
    cost_obs = normalize_cost_matrix(cost_obs)

    total_cost = alpha * cost_trans + (1 - alpha) * cost_obs
    row_ind, col_ind = linear_sum_assignment(total_cost)

    # permutation_order[true_index] = learned_index
    permutation_order = np.zeros(n_states, dtype=int)
    permutation_order[col_ind] = row_ind

    # perm_map_dict[true_index] = learned_index
    perm_map_dict = {int(c): int(r) for r, c in zip(row_ind, col_ind)}

    return permutation_order, perm_map_dict


def compute_error_metrics(learned_model, true_transitions, true_observations, states, dist_type="beta"):
    """
    Compute error metrics between the learned model and the true model.
    :param learned_model: The learned model with transition and observation parameters
    :param true_transitions: The true transition parameters.
    :param true_observations: The true observation parameters.
    :param states: List of state names.
    :param dist_type: Type of distribution for observations ("beta" or "mvn").
    :return: A dictionary containing final KL divergences, average L1 error, and permutation map.
    """
    n_states = learned_model.n_states
    n_actions = learned_model.n_actions

    perm_ord, perm_map = match_state_hungarian(learned_model, true_transitions, true_observations, states,
                                               dist_type=dist_type, alpha=0.1)
    aligned_means = learned_model.obs_means[perm_ord]
    aligned_covs = learned_model.obs_covs[perm_ord]

    # Final KL
    final_kl = np.zeros(n_states)
    for state in range(n_states):
        learned_obs_mvn = _from_mean_cov_to_gmm(aligned_means[state], aligned_covs[state])
        true_obs_mvn = _from_dist_to_mvn(true_observations, dist_type, state_name=states[state])

        KL_row_col = _compute_KL_divergence(true_obs_mvn, learned_obs_mvn)
        KL_col_row = _compute_KL_divergence(learned_obs_mvn, true_obs_mvn)
        final_kl[state] = 0.5 * (KL_row_col + KL_col_row)

    aligned_transitions = learned_model.transitions[perm_ord][:, :, perm_ord]
    l1_diff = np.sum(np.abs(aligned_transitions - true_transitions))
    avg_l1_error = l1_diff / (n_states * n_actions)

    return {
        "final_kl": final_kl,
        "avg_l1_error": avg_l1_error,
        "perm_map": perm_map,
        "perm_ord": perm_ord
    }


def visualize_L1_trials(results, noise_level=0.01, env_name='', folder_name=''):
    sns.set_theme(style="whitegrid", context="talk",
                  rc={
                      "grid.color": ".9",
                      "grid.linewidth": 1.0,
                      "axes.edgecolor": ".3",
                      "axes.linewidth": 0.8,
                  }
                  )
    plt.figure(figsize=(10, 6))
    plot_data = []

    for model_name, trials in results.items():
        for trial in trials:
            plot_data.append({
                'Model': model_name,
                'Data Size': trial['data_size'],
                'L1 Error': trial['metrics']['avg_l1_error'],
            })
        datasizes = set([trail['data_size'] for trail in trials])
        for datasize in datasizes:
            L1_values = np.array(
                [trial['metrics']['avg_l1_error'] for trial in trials if trial['data_size'] == datasize])
            L1_mean = float(np.mean(L1_values))
            L1_sd = float(np.std(L1_values, ddof=1)) if L1_values.size > 1 else 0.0
            count = int(L1_values.size)
            print(
                f"{model_name} | data_size={datasize} | mean={L1_mean:.6f} | sd={L1_sd:.6f} | n={count}")

    df = pd.DataFrame(plot_data)
    #sns.set_theme()
    plot_obj = sns.lineplot(
        data=df,
        x='Data Size',
        y='L1 Error',
        hue='Model',
        marker='o',
        linewidth=2,
        style='Model'
    )
    _vmin, _vmax = _compute_lim_from_ci(plot_obj, margin=0.1, vmin=0, vmax=1.2)
    plt.ylim([_vmin, _vmax])
    plt.title('Impact of Data Size on Model Error (L1) - Noise Level: ' + str(noise_level))
    plt.xlabel('Data Size')
    plt.ylabel('Average L1 Error')
    plt.legend(title='Model Type',
               frameon=True,
               loc='upper right',
               framealpha=0.8
               )
    path = folder_name + "avg_l1_error_" + env_name + '_SD' + str(noise_level) + '.png'
    plt.savefig(path, bbox_inches='tight', pad_inches=0.05, dpi=300)
    plt.tight_layout()
    plt.show()


def visualize_KL_trials(results, noise_level='', env_name='',  folder_name=''):
    sns.set_theme(style="whitegrid", context="talk",
                  rc={
                      "grid.color": ".9",
                      "grid.linewidth": 1.0,
                      "axes.edgecolor": ".3",
                      "axes.linewidth": 0.8,
                  }
                  )
    plt.figure(figsize=(10, 6))
    plot_data = []

    for model_name, trials in results.items():
        for trial in trials:
            KL_values = np.mean(trial['metrics']['final_kl'])
            plot_data.append({
                'Model': model_name,
                'Data Size': trial['data_size'],
                'KL Error': KL_values,
            })

        datasizes = set([trail['data_size'] for trail in trials])
        for datasize in datasizes:
            KL_values = np.array(
                [np.mean(trial['metrics']['final_kl']) for trial in trials if trial['data_size'] == datasize])
            KL_mean = float(np.mean(KL_values))
            KL_sd = float(np.std(KL_values, ddof=1)) if KL_values.size > 1 else 0.0
            count = int(KL_values.size)
            print(
                f"{model_name} | data_size={datasize} | mean={KL_mean:.6f} | sd={KL_sd:.6f} | n={count}")

    df = pd.DataFrame(plot_data)

    #sns.set_theme()
    plot_obj = sns.lineplot(
        data=df,
        x='Data Size',
        y='KL Error',
        hue='Model',
        marker='o',
        linewidth=2,
        style='Model',
    )
    _vmin, _vmax = _compute_lim_from_ci(plot_obj, margin=0.1, vmin=0, vmax=12.0)
    plt.ylim([_vmin, _vmax])
    plt.title('Impact of Data Size on KL Divergence -  Noise Level: ' + str(noise_level))
    plt.xlabel('Data Size')
    plt.ylabel('Average KL divergence')
    plt.legend(title='Model Type',
               frameon=True,
               loc='upper right',
               framealpha=0.8)
    path =  path = folder_name + "KL_divergence_" + env_name + '_SD' + str(noise_level) + '.png'
    plt.savefig(path, bbox_inches='tight', pad_inches=0.05, dpi=300)
    plt.tight_layout()
    plt.show()


def plot_state_observation_distributions(model, feature_names=None, save_path=None):
    """
    Plots the Gaussian observation distributions for each hidden state in the POMDP.
    Creates one figure with subplots for each observation dimension.
    
    :param model: The trained POMDP model (PomdpEM or FuzzyMAP_EM)
    :param feature_names: List of strings for the axes labels (e.g., ['T1', 'T2', 'V', 'E'])
    :param save_path: Optional path to save the generated figure
    """
    # Ensure arrays are numpy formats
    obs_means = np.array(model.obs_means)
    obs_covs = np.array(model.obs_covs)
    
    n_states = obs_means.shape[0]
    obs_dim = obs_means.shape[1]
    
    # Fallback if no feature names are provided
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(obs_dim)]
        
    # Set up the figure: 1 row, 'obs_dim' columns
    fig, axes = plt.subplots(1, obs_dim, figsize=(4 * obs_dim, 4.5))
    if obs_dim == 1:
        axes = [axes]
        
    # Use a distinct color palette for the states
    colors = plt.cm.get_cmap('tab10', n_states)

    for obs_idx in range(obs_dim):
        ax = axes[obs_idx]
        
        # 1. Find a sensible X-axis range for this specific feature
        # Look at all state means for this feature and add/subtract 3 standard deviations
        stds = [np.sqrt(max(obs_covs[s, obs_idx, obs_idx], 1e-9)) for s in range(n_states)]
        x_min = np.min(obs_means[:, obs_idx]) - 3 * max(stds)
        x_max = np.max(obs_means[:, obs_idx]) + 3 * max(stds)
        
        x = np.linspace(x_min, x_max, 500)
        
        # 2. Plot the Gaussian curve for each hidden state
        for state in range(n_states):
            mean = obs_means[state, obs_idx]
            variance = obs_covs[state, obs_idx, obs_idx]
            std_dev = np.sqrt(max(variance, 1e-9))  # Guard against negative/zero variance
            
            # Generate Probability Density Function
            pdf = norm.pdf(x, loc=mean, scale=std_dev)
            
            # Plot line and fill area under the curve
            ax.plot(x, pdf, label=f'State {state}', color=colors(state), lw=2)
            ax.fill_between(x, pdf, alpha=0.1, color=colors(state))
            
        ax.set_title(f'{feature_names[obs_idx]} Emission Model')
        ax.set_xlabel('Value (z-score)')
        ax.set_ylabel('Probability Density')
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Only put the legend on the last subplot to save space
        if obs_idx == obs_dim - 1:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Hidden States")

    plt.suptitle('POMDP Hidden State Observation Distributions', fontsize=14, y=1.05)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Distributions plot saved to {save_path}")
        
    #plt.show()

def plot_grid_search_heatmap(experiment_results, metric_key='final_kl',
                             param1='lambda_T', param2='lambda_O',
                             title="Hyperparameter Grid Search",
                             exp_name="", folder_name="res/",
                             vmax=None):
    sns.set_theme()
    data = []
    model_name = None
    for model_name, trails in experiment_results.items():
        p1_match = re.search(f"{param1}=([0-9.]+)", model_name)
        p2_match = re.search(f"{param2}=([0-9.]+)", model_name)

        val1 = float(p1_match.group(1))
        val2 = float(p2_match.group(1))

        values = []
        for trail in trails:
            metric_values = trail['metrics'][metric_key]
            if np.ndim(metric_values) > 0:
                values.append(np.mean(metric_values))
            else:
                values.append(metric_values)

        avg_score = np.mean(values)
        data.append({param1: val1, param2: val2, metric_key: avg_score})

    env_name = experiment_results[model_name][0]["env_name"]
    df = pd.DataFrame(data)

    # Sort the data numerically
    df = df.sort_values(by=[param1, param2], ascending=[False, True])

    # Pivot the table
    pivot_table = df.pivot(index=param1, columns=param2, values=metric_key)

    pivot_table.sort_index(axis=0, ascending=False, inplace=True)
    pivot_table.sort_index(axis=1, ascending=True, inplace=True)

    plt.figure(figsize=(10, 8))

    # Plot
    _vmin, _vmax = _compute_lim_from_values(pivot_table.values.flatten(), margin=0.1, vmax=vmax, vmin=0)
    ax = sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="viridis_r", vmax=_vmax, vmin=_vmin)

    # Format the tick labels
    ax.set_yticklabels([f"{y:.2f}" for y in pivot_table.index], rotation=0)
    ax.set_xticklabels([f"{x:.2f}" for x in pivot_table.columns], rotation=45)

    ax.set_ylabel(param1)
    ax.set_xlabel(param2)

    plt.title(title)
    plt.tight_layout()
    plt.savefig(folder_name + metric_key + "_" + env_name + "_grid_search_heatmap.png",
                bbox_inches='tight', pad_inches=0.01)
    plt.show()


def plot_1d_sensitivity(experiment_results, param_name='alpha_ah',
                        metric_key='kl_final', folder_name="res/",
                        title="Parameter Sensitivity", vmax=None, vmin=0):
    data = []
    model_name = None
    for model_name, trails in experiment_results.items():
        p_match = re.search(f"{param_name}=([0-9.]+)", model_name)
        alpha_val = p_match.group(1)

        for trail in trails:
            val = trail['metrics'][metric_key]
            if np.ndim(val) > 0:
                val = np.mean(val)

            data.append({param_name: alpha_val, 'score': val})

    env_name = experiment_results[model_name][0]["env_name"]
    df = pd.DataFrame(data)

    plt.figure(figsize=(10, 6))

    plot_obj = sns.lineplot(data=df, x=param_name, y='score', marker='o', linewidth=2)

    _vmin, _vmax = _compute_lim_from_ci(plot_obj, margin=0.1, vmax=vmax, vmin=vmin)
    plt.ylim([_vmin, _vmax])
    plt.title(title)
    plt.xlabel(param_name)
    plt.ylabel(metric_key)
    plt.tight_layout()
    plt.savefig(folder_name + metric_key + "_" + env_name + "_alpha_ah_sensitivity_plot.png",
                bbox_inches='tight', pad_inches=0.01)
    plt.show()


def _compute_lim_from_values(values, margin=0.1, vmax=None, vmin=None):
    min_val = np.min(values)
    max_val = np.max(values)
    y_min = min_val - margin * abs(min_val)
    y_max = max_val + margin * abs(max_val)
    if vmax is not None:
        y_max = min(y_max, vmax)
    if vmin is not None:
        y_min = max(y_min, vmin)

    return y_min, y_max


def _compute_lim_from_ci(plot_obj, margin=0.1, vmax=None, vmin=None):
    all_vertices = np.empty((0, 2))
    if len(plot_obj.collections) > 1:
        for coll in plot_obj.collections:
            vertices = coll.get_paths()[0].vertices
            all_vertices = np.concatenate((all_vertices, vertices), axis=0)
    else:
        all_vertices = plot_obj.collections[0].get_paths()[0].vertices
    y_min, y_max = _compute_lim_from_values(all_vertices[:, 1], margin=margin, vmax=vmax, vmin=vmin)

    return y_min, y_max

def _parse_trajectory(trajectory):
    """
    Helper to parse a flat trajectory list [o_0, a_0, o_1, a_1, o_2] 
    into separate observation and action sequences expected by PomdpEM.
    """
    # If the trajectory is already split into a tuple, return it directly
    if isinstance(trajectory, tuple) and len(trajectory) == 2:
        return np.array(trajectory[0]), np.array(trajectory[1])
        
    obs_seq = []
    act_seq = []
    for i, item in enumerate(trajectory):
        if i % 2 == 0:
            obs_seq.append(item)
        else:
            act_seq.append(int(item))
            
    return np.array(obs_seq), np.array(act_seq)


def compute_log_likelihood(model, obs_seqs, act_seqs):
    """
    Computes the average predictive log-likelihood of a held-out test set.
    
    :param model: The trained POMDP model (PomdpEM or FuzzyMAP_EM)
    :param obs_seqs: List of observation sequences
    :param act_seqs: List of action sequences
    :return: Average log-likelihood per trajectory
    """
    total_ll = 0.0
    
    for i in range(len(obs_seqs)):
        obs_seq = obs_seqs[i]
        act_seq = act_seqs[i]
        
        # Use the model's internal methods to compute the emission probabilities P(o|s)
        obs_probs = model._compute_emission_matrix(obs_seq)
        
        # The forward pass returns the filtered belief states (alpha), 
        # the log likelihood of the sequence, and the scaling factors (c)
        _, seq_ll, _ = model.forward_pass(obs_probs, act_seq)
        
        total_ll += seq_ll
        
    # Return the average log-likelihood across all test trajectories
    return total_ll / len(obs_seqs) if len(obs_seqs) > 0 else 0.0


def compute_avg_l1_error(model, obs_seqs, act_seqs):
    """
    Computes the one-step-ahead observation L1 error on a test set.
    Evaluates how accurately the model predicts the continuous vital signs at t+1.
    
    :param model: The trained POMDP model
    :param obs_seqs: List of observation sequences
    :param act_seqs: List of action sequences
    :return: Average L1 error across all prediction steps
    """
    total_l1 = 0.0
    count = 0
    
    for i in range(len(obs_seqs)):
        obs_seq = obs_seqs[i]
        act_seq = act_seqs[i]
        
        # Compute emissions and run forward pass to extract belief states
        obs_probs = model._compute_emission_matrix(obs_seq)
        alpha, _, _ = model.forward_pass(obs_probs, act_seq)
        
        # alpha[t] represents the normalized belief state b_t
        for t in range(len(act_seq)):
            b_t = alpha[t]
            action = act_seq[t]
            actual_next_obs = obs_seq[t+1]
            
            # 1. Predict the next state distribution: P(s_{t+1} | b_t, a_t)
            # model.transitions is shape (n_states, n_actions, n_states) -> P(s' | s, a)
            trans_matrix = model.transitions[:, action, :] 
            next_state_probs = b_t @ trans_matrix
            
            # 2. Predict the next continuous observation: E[o_{t+1} | s_{t+1}]
            # Because the emissions are Gaussian, the expected value is the weighted mean
            expected_obs = next_state_probs @ model.obs_means
            
            # 3. Calculate L1 error (Mean Absolute Error across the continuous dimensions)
            l1 = np.mean(np.abs(expected_obs - actual_next_obs))
            total_l1 += l1
            count += 1
            
    return total_l1 / count if count > 0 else 0.0
