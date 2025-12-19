import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import multivariate_normal


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
            # Observation KL divergence: learned (row) vs true (col)
            learned_obs_mvn = _mvn_definition_from_POMDP(learned_model, state=row)
            true_obs_mvn = _from_dist_to_mvn(true_observations, dist_type, state_name=states[col])

            KL_row_col = _compute_KL_divergence(learned_obs_mvn, true_obs_mvn)
            KL_col_row = _compute_KL_divergence(true_obs_mvn, learned_obs_mvn)
            cost_obs[row, col] = 0.5 * (KL_row_col + KL_col_row)

            # Transition L1 distance: learned (row) vs true (col)
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

    #print the true mean and cov
    #TODO: remove this print statement later
    #print("True Means and Covariances:")
    #for state in range(n_states):
    #    true_obs_mvn = _from_dist_to_mvn(true_observations, dist_type, state_name=states[state])
    #    print(f"State {state}: Mean = {true_obs_mvn.mean}, Covariance = {true_obs_mvn.cov}")

    return {
        "final_kl": final_kl,
        "avg_l1_error": avg_l1_error,
        "perm_map": perm_map,
        "perm_ord": perm_ord
    }
