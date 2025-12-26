import numpy as np

from models.trainable.pomdp_EM import PomdpEM


class PomdpMAPEM(PomdpEM):
    """
    POMDP Expectation-Maximization with Maximum A Posteriori (MAP) estimation.
    This class extends the PomdpEM class to include MAP estimation for the model parameters.
    This class is intended for non-informative priors.
    """

    def __init__(
            self, n_states,
            n_actions,
            obs_dim,
            verbose: bool = False,
            seed: int = 42,
            parallel: bool = True,
            prior_mu0=0.0,
            prior_kappa0=1.0,
            prior_psi0=1.0,
            prior_nu0=None,
            prior_alpha=1.0,
    ):
        super().__init__(
            n_states,
            n_actions,
            obs_dim,
            verbose,
            seed,
            parallel
        )

        self.prior_alpha = prior_alpha

        self.prior_mu0 = np.full(self.obs_dim, prior_mu0)
        self.prior_kappa0 = float(prior_kappa0)
        self.prior_psi0 = np.eye(self.obs_dim) * float(prior_psi0)
        self.prior_nu0 = (self.obs_dim + 2) if prior_nu0 is None else prior_nu0

    def maximization_step(self, observations, actions, gammas, xis, iteration=0):
        # Number of sequences
        n_observations = len(observations)

        emp_N_T = np.zeros_like(self.transitions)
        emp_den_T = np.zeros((self.n_states, self.n_actions))

        emp_N_O = np.zeros(self.n_states)
        emp_Sum_O = np.zeros((self.n_states, self.obs_dim))
        emp_Sum_sq_O = np.zeros((self.n_states, self.obs_dim, self.obs_dim))

        for i in range(n_observations):
            obs_np = np.array(observations[i])
            actions_np = np.array(actions[i])
            for a in range(self.n_actions):

                t_indices = np.where(actions_np[:-1] == a)[0]
                if len(t_indices) > 0:
                    emp_N_T[:, a, :] += np.sum(xis[i][a], axis=0)
                    emp_den_T[:, a] += np.sum(xis[i][a], axis=(0, 1))

            for s in range(self.n_states):
                gamma_s = gammas[i][:, s]
                emp_N_O[s] += np.sum(gamma_s)
                emp_Sum_O[s] += gamma_s @ obs_np
                emp_Sum_sq_O[s] += (obs_np.T * gamma_s) @ obs_np

        # Transitions

        num = emp_N_T + self.prior_alpha - 1.0
        den = np.sum(num, axis=2, keepdims=True)
        den[den == 0] = 1.0  # To avoid division by zero
        self.transitions = num / den

        # Observations
        for s in range(self.n_states):
            if emp_N_O[s] < 1e-10:
                continue
            emp_mean = emp_Sum_O[s] / emp_N_O[s]
            scatter_matrix = emp_Sum_sq_O[s] - emp_N_O[s] * np.outer(emp_mean, emp_mean)
            diff_means = emp_mean - self.prior_mu0

            kappa_n = self.prior_kappa0 + emp_N_O[s]
            nu_n = self.prior_nu0 + emp_N_O[s]
            mu_n = (self.prior_kappa0 * self.prior_mu0 + emp_Sum_O[s]) / kappa_n
            psi_n = (self.prior_psi0 + scatter_matrix
                     + (self.prior_kappa0 * emp_N_O[s] / kappa_n) * np.outer(diff_means, diff_means))

            self.obs_means[s] = mu_n

            # Ensure covariance matrix is positive definite
            self.obs_covs[s] = psi_n / (nu_n + self.obs_dim + 1)

        self.initial_prob = np.mean([g[0] for g in gammas], axis=0)
        return
