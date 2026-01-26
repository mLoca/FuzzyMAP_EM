import re

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import seaborn as sns


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
    plt.savefig(path, bbox_inches='tight', pad_inches=0.05)
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
    plt.savefig(path, bbox_inches='tight', pad_inches=0.05)
    plt.tight_layout()
    plt.show()


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
