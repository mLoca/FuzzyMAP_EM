import torch
import numpy as np
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, TraceEnum_ELBO
from pyro.infer.autoguide import AutoDelta
from pyro.optim import Adam
import scipy.stats
import numpy as np

class VariationalIOHMM:
    """
    Modern Bayesian Baseline: Variational Input-Output HMM.
    Uses Stochastic Variational Inference (SVI) with exact discrete latent marginalization.
    """
    def __init__(self, n_states, n_actions, obs_dim):
        self.n_states = n_states
        self.n_actions = n_actions
        self.obs_dim = obs_dim
        
        self.transitions = None
        self.obs_means = None
        self.obs_covs = None

    def model(self, obs_batch, acts_batch):
        N, T_plus_1, _ = obs_batch.shape
        T = T_plus_1 - 1
        
        pi_logits = pyro.param("pi_logits", torch.zeros(self.n_states))
        pi = torch.softmax(pi_logits, dim=-1)
        
        trans_logits = pyro.param("trans_logits", torch.zeros(self.n_actions, self.n_states, self.n_states))
        trans = torch.softmax(trans_logits, dim=-1)
        
        obs_mu = pyro.param("obs_mu", torch.randn(self.n_states, self.obs_dim))
        obs_scale_tril = pyro.param("obs_scale_tril", 
                                    torch.eye(self.obs_dim).repeat(self.n_states, 1, 1),
                                    constraint=dist.constraints.lower_cholesky)

        with pyro.plate("batch", N):
            z = pyro.sample("z_0", dist.Categorical(pi), infer={"enumerate": "parallel"})
            pyro.sample("obs_0", dist.MultivariateNormal(obs_mu[z], scale_tril=obs_scale_tril[z]), obs=obs_batch[:, 0])
            
            for t in pyro.markov(range(T)):
                a_t = acts_batch[:, t]
                z = pyro.sample(f"z_{t+1}", dist.Categorical(trans[a_t, z]), infer={"enumerate": "parallel"})
                pyro.sample(f"obs_{t+1}", dist.MultivariateNormal(obs_mu[z], scale_tril=obs_scale_tril[z]), obs=obs_batch[:, t+1])

    def fit(self, observations, actions, max_iterations=500, tolerance=1e-4):
        pyro.clear_param_store()
        
        obs_batch = torch.tensor(np.array(observations), dtype=torch.float32)
        acts_batch = torch.tensor(np.array(actions), dtype=torch.long)
        
        guide = AutoDelta(pyro.poutine.block(self.model, expose=["pi_logits", "trans_logits", "obs_mu", "obs_scale_tril"]))
        optim = Adam({"lr": 0.05})
        svi = SVI(self.model, guide, optim, loss=TraceEnum_ELBO(max_plate_nesting=1))
        
        prev_loss = float('inf')
        for epoch in range(max_iterations):
            loss = svi.step(obs_batch, acts_batch)
            if abs(prev_loss - loss) < tolerance:
                break
            prev_loss = loss
            
        trans_action_first = torch.softmax(pyro.param("trans_logits"), dim=-1).detach().numpy()
        # Reshape from (actions, states, states) to (states, actions, states)
        self.transitions = np.transpose(trans_action_first, (1, 0, 2))
        self.obs_means = pyro.param("obs_mu").detach().numpy()
        
        L = pyro.param("obs_scale_tril").detach()
        self.obs_covs = torch.matmul(L, L.transpose(-1, -2)).numpy()
        self.initial_state_dist = torch.softmax(pyro.param("pi_logits"), dim=-1).detach().numpy()

    def _compute_emission_matrix(self, obs_seq):
        """
        Computes P(o_t | s_t) for all timesteps and states.
        Expected by utils.metrics.py
        """
        T = len(obs_seq)
        obs_probs = np.zeros((T, self.n_states))
        
        for state in range(self.n_states):
            # Evaluate the multivariate normal PDF for this state across all timesteps
            mvn = scipy.stats.multivariate_normal(mean=self.obs_means[state], cov=self.obs_covs[state])
            obs_probs[:, state] = mvn.pdf(obs_seq)
            
        # Add a tiny epsilon to prevent log(0) issues in the forward pass
        return np.clip(obs_probs, 1e-100, None)

    def forward_pass(self, obs_probs, act_seq):
        """
        Standard HMM Forward Algorithm to compute belief states (alpha) and log-likelihood.
        Expected by utils.metrics.py
        """
        T = len(obs_probs)
        alpha = np.zeros((T, self.n_states))
        c = np.zeros(T)

        # Time t=0
        alpha[0] = self.initial_state_dist * obs_probs[0]
        c[0] = np.sum(alpha[0])
        if c[0] == 0: c[0] = 1e-100
        alpha[0] /= c[0]

        # Time t > 0
        for t in range(1, T):
            action = int(act_seq[t-1])
            
            # Predict next belief: P(s_t | o_{1:t-1}, a_{1:t-1})
            # T is shape (n_actions, prev_state, next_state)
            pred_belief = alpha[t-1] @ self.transitions[:, action, :]
            
            # Update with observation: P(s_t | o_{1:t}, a_{1:t-1})
            alpha[t] = pred_belief * obs_probs[t]
            c[t] = np.sum(alpha[t])
            
            # Scale to prevent underflow
            if c[t] == 0: c[t] = 1e-100
            alpha[t] /= c[t]

        # Sequence log likelihood is the sum of the log of the scaling factors
        seq_ll = np.sum(np.log(c))
        
        return alpha, seq_ll, c 