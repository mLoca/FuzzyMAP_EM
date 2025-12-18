import numpy as np
from scipy.optimize import linear_sum_assignment


def _from_beta_to_multivariate_normal(obs_model_real, state_name):
    """
    Convert beta parameters to a multivariate normal distribution.
    :param obs_model_real: The observation model containing beta parameters.
    :param state_name: The name of the state for which to convert the parameters.
    :return: A GaussianMixture object representing the multivariate normal distribution.
    """
    from sklearn.mixture import GaussianMixture

    means = []
    variances = []
    for dim, p in obs_model_real[0][state_name].items():
        a, b = p
        # Beta mean and variance
        means.append(a / (a + b))
        variances.append((a * b) / ((a + b) ** 2 * (a + b + 1)))

    real_GMM = GaussianMixture(n_components=1, means_init=means)
    real_GMM.means_ = np.array([means])
    real_GMM.weights_ = [1]  # Assuming equal weights for simplicity
    real_GMM.covariances_ = [np.diag(variances), np.diag(variances)]

    return real_GMM


def _compute_KL_divergence(gmm_p, gmm_q):
    """
    Compute the KL divergence between two Gaussian Mixture Models.
    :param gmm_p: The first Gaussian Mixture Model (GMM).
    :param gmm_q: The second Gaussian Mixture Model (GMM).
    :return: Closed-form KL divergence D_KL(P \| Q) for single-component Gaussian GMMs.
    """
    k = gmm_p.means_.shape[1]

    cov0 = gmm_p.covariances_[0]
    cov1 = gmm_q.covariances_[0]
    cov0 += 1e-6 * np.eye(cov0.shape[0])
    cov1 += 1e-6 * np.eye(cov1.shape[0])

    cov1_inv = np.linalg.inv(cov1)

    trace_term = np.trace(cov1_inv @ cov0)
    diff = gmm_q.means_ - gmm_p.means_
    term = diff @ cov1_inv @ diff.T

    (sign0, logdet0) = np.linalg.slogdet(cov0)
    (sign1, logdet1) = np.linalg.slogdet(cov1)
    log_det_term = logdet1 - logdet0
    kl_approx = 0.5 * (trace_term + term + log_det_term - k)
    return kl_approx


def _from_mvn_to_gmm(mvns, state):
    """
    Convert a multivariate normal distribution to a Gaussian Mixture Model.
    :param mvns: A set of MVN representing the multivariate normal distribution in pomdp.
    :param state: The  state to convert.
    :return: A GaussianMixture object.
    """
    from sklearn.mixture import GaussianMixture

    gmm = GaussianMixture(n_components=1, means_init=mvns.obs_means[state])
    gmm.means_ = mvns.obs_means[state].reshape(1, -1)
    gmm.weights_ = [1]  # Assuming equal weights for simplicity
    gmm.covariances_ = [mvns.obs_covs[state], mvns.obs_covs[state]]

    return gmm


def _from_mean_cov_to_gmm(means, cov):
    """
    Convert a multivariate normal distribution to a Gaussian Mixture Model.
    :param mvns: A set of MVN representing the multivariate normal distribution in pomdp.
    :param state: The  state to convert.
    :return: A GaussianMixture object.
    """
    from sklearn.mixture import GaussianMixture

    gmm = GaussianMixture(n_components=1, means_init=means)
    gmm.means_ = means.reshape(1, -1)
    gmm.weights_ = [1]  # Assuming equal weights for simplicity
    gmm.covariances_ = [cov, cov]

    return gmm


def normalize_cost_matrix(cost_matrix):
    if cost_matrix.max() == cost_matrix.min(): return np.zeros_like(cost_matrix)
    return (cost_matrix - cost_matrix.min()) / (cost_matrix.max() - cost_matrix.min())


def match_state_hungarian(learned_model, true_transitions, true_observations, states, alpha=0.35):
        """
        Match the states of the learned model to the true model using the Hungarian algorithm.
        :param learned_model: The learned model with transition and observation parameters
        :param true_transitions: The true transition parameters.
        :param true_observations: The true observation parameters.
        :param alpha: Weighting factor between transition and observation KL divergences.
        :param states: List of state names.
        :return: A list of matched state indices.
        """
        n_states = learned_model.n_states

        cost_trans = np.zeros((n_states, n_states))
        cost_obs = np.zeros((n_states, n_states))
        for row in range(n_states):
            for col in range(n_states):
                # Observation KL divergence: learned (row) vs true (col)
                learned_obs_gmm = _from_mvn_to_gmm(learned_model, state=row)
                true_obs_gmm = _from_beta_to_multivariate_normal(true_observations, state_name=states[col])

                KL_row_col = _compute_KL_divergence(learned_obs_gmm, true_obs_gmm)
                KL_col_row = _compute_KL_divergence(true_obs_gmm, learned_obs_gmm)
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


def compute_error_metrics(learned_model, true_transitions, true_observations, states):

    """
    Compute error metrics between the learned model and the true model.
    :param learned_model: The learned model with transition and observation parameters
    :param true_transitions: The true transition parameters.
    :param true_observations: The true observation parameters.
    :param states: List of state names.
    :return: A dictionary containing final KL divergences, average L1 error, and permutation map.
    """
    n_states = learned_model.n_states
    n_actions = learned_model.n_actions

    perm_ord, perm_map = match_state_hungarian(learned_model, true_transitions, true_observations, states, alpha=0.1)
    aligned_means = learned_model.obs_means[perm_ord]
    aligned_covs = learned_model.obs_covs[perm_ord]

    # Final KL
    final_kl = np.zeros(n_states)
    for state in range(n_states):
        learned_obs_gmm = _from_mean_cov_to_gmm(aligned_means[state], aligned_covs[state])
        true_obs_gmm = _from_beta_to_multivariate_normal(true_observations, state_name=states[state])

        KL_row_col = _compute_KL_divergence(true_obs_gmm, learned_obs_gmm)
        KL_col_row = _compute_KL_divergence(learned_obs_gmm, true_obs_gmm)
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
