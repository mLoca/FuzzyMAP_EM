import json
import os
import sys
from contextlib import contextmanager

import numpy as np
from scipy.special import digamma


def _load_pomdp_config(config_path):
    with open(config_path, 'r') as file:
        config = json.load(file)
    return config


def from_matrix_to_triplelist(transition_matrix, states=None, actions=None):
    triples = {}
    for j, action in enumerate(actions):
        for i, state in enumerate(states):
            for k, next_state in enumerate(states):
                if transition_matrix[i][j][k] > -0.01:
                    triples.update({
                        (action, state, next_state): transition_matrix[i][j][k]
                    })
                else:
                    # throw an error if the transition probability is negative
                    raise ValueError(f"Transition probability for action {action}, state {state}, "
                                     f"next state {next_state} is negative: {transition_matrix[i][j][k]}")
    return triples


def from_observation_config_to_observation(observation_config, states=None, obs_names=None, distribution_type="mvn"):
    obs = {}
    for state in states:
        obs.update({
            state: {}
        })
        for obs_name in obs_names:
            obs[state].update({
                obs_name: []
            })
            obs[state][obs_name].extend(observation_config[0][state][obs_name])

    return obs


def from_beta_to_multivariate_normal(beta):
    """
    Convert beta parameters to a multivariate normal distribution.
    :param beta: A list or array of beta parameters.
    :return: A dictionary representing the multivariate normal distribution.
    """
    means = np.zeros((len(beta), len(list(beta.values())[0]) // 2))
    covariances = np.zeros((len(beta),
                            len(list(beta.values())[0]) // 2,
                            len(list(beta.values())[0]) // 2))

    count = 0
    for state, val in beta.items():
        a_t, b_t, a_s, b_s = beta[state]
        # Beta mean and variance
        mean_t = a_t / (a_t + b_t)
        mean_s = a_s / (a_s + b_s)
        var_t = (a_t * b_t) / ((a_t + b_t) ** 2 * (a_t + b_t + 1))
        var_s = (a_s * b_s) / ((a_s + b_s) ** 2 * (a_s + b_s + 1))

        means[count, :] = [mean_t, mean_s]
        covariances[count, :, :] = [[var_t, 0],
                                    [0, var_s]]
        count += 1

    return {
        "means": means,
        "covariances": covariances
    }


def _prob_sum(p, q):
    return p + q - p * q

@staticmethod
def multidigamma(x, d):
    """Compute the multivariate digamma function"""
    result = 0
    for i in range(1, d + 1):
        result += digamma(x + (1 - i) / 2)
    return result


@contextmanager
def suppress_output():
    """
    Context manager to temporarily suppress stdout output.
    """
    # Save the original stdout
    original_stdout = sys.stdout
    # Redirect stdout to devnull
    sys.stdout = open(os.devnull, 'w')

    try:
        yield
    finally:
        # Restore stdout
        sys.stdout.close()
        sys.stdout = original_stdout
