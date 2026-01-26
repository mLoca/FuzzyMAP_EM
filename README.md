# Fuzzy-MAP EM
This repository contains the implementation of the Fuzzy-MAP EM algorithm presented in the paper [Fuzzy-MAP EM](https://arxiv.org/abs/2511.14619).

### About the Paper
Learning Partially Observable Markov Decision Processes (POMDPs) in healthcare is often hindered by data scarcity and high noise. This work introduces a novel approach that integrates expert knowledge into the learning process. By encoding expert insights into a Fuzzy Logic model (Type-1 Takagi-Sugeno), the algorithm generates "fuzzy pseudo-counts" that act as informative priors during the Expectation-Maximization (EM) process. This effectively converts the problem into a Maximum A Posteriori (MAP) estimation, significantly improving model recovery in low-data regimes.

## Installation and Requirements
### Requirements
The project relies on the following Python libraries:
```txt
numpy>=1.24
scipy>=1.10
pyfume>=0.1.0
PyYAML>=6.0
matplotlib>=3.7
seaborn>=0.12
tqdm>=4.65
pandas>=2.0
scikit-learn>=1.2
```
### Installation
```bash
pip install -r requirements.txt
```

## 1. Synthetic Experiments Guide
The synthetic experiments compare the Fuzzy-MAP EM against standard baselines (Standard EM, Standard MAP) on simulated medical environments.

### Running the experiemnts
To reproduce the synthetic experiments, execute the main script:
```bash
python experiments.py
```
This script reads configurations from config/experiments.yaml, generates synthetic data, trains the specified models, and saves results to res/

### Creating a New Experiment 
The system is configured via the `config/experiments.yaml` file. To create a new experiment:
- Define the Environment: Under the `environments` key, add a new entry. You must specify the states, actions, observations, and ground truth probabilities.
  ```yaml
  - name: "my_custom_env"
    n_states: 3
    n_actions: 2
    obs_dim: 2  # Dimensionality of the observation vector
    states: ["healthy", "sick", "critical"]
    ...
  ```
- Define the Experiment: Under experiments, create a new block (e.g., my_experiment_1), set ```active: true```,
write the environment in the environment list of the experiment ```environments: ["my_env"]```, and specify other parameters such as  `dataset_sizes, noise_level, and n_trials`.
> **⚠️ Important:** Changing Dimensions  
If your new experiment requires a different number of actions or observations (`obs_dim`) than the defaults, you must modify the fuzzy model generation logic in `fuzzy/fuzzy_model.py`. Specifically, update the input dimension of the fuzzy model to match your new observation space.
- Run the Experiment: Execute the `experiments.py` script to run your new experiment.

## 2. Myastheniia Gravis  Experiments
The `MG/` folder contains the code for the real-world case study on Myasthenia Gravis.
### Option A: One POMDP Retrieval
To perform a single run of the algorithm to recover the POMDP parameters  from the MG fuzzy model simulation:
``` bash
python MG/fuzzy_EM.py
```
This script initializes the Fuzzy-MAP EM learner with the specific MG expert model and visualizes the resulting observation distributions.
### Option B: Bootstrap Analysis
To conduct a bootstrap analysis for robustness evaluation:
``` bash
python MG/bootstrap_analysis.py
```
This script performs multiple runs of the Fuzzy-MAP EM algorithm on bootstrapped datasets, collecting statistics on the recovered parameters.

## 3. Custom Experiments with External Data
You can use the `FuzzyPOMDP` class to train models on your own custom datasets using your own expert fuzzy model.
Prerequisites
* Data: Must be formatted as a list of sequences (numpy arrays).
* Fuzzy Model: Must be a pyfume.FuzzySystem object (using the pyFUME library).

### Example Code
```python
import numpy as np
from models.trainable.fuzzy_EM import FuzzyPOMDP
# Import your own fuzzy model builder (must return a pyfume.FuzzySystem)
from my_custom_fuzzy_logic import build_my_expert_model 

# Prepare the data
observations = [ [obs_1_t0, obs_1_t1, ...], [obs_2_t0, ...], ... ]
actions = [ [act_1_t0, act_1_t1, ...], ... ]

#Load Expert Model
fuzzy_expert = build_my_expert_model()

# initialize Fuzzy-MAP EM
model = FuzzyPOMDP(
    n_states=3,                   # Desired latent states
    n_actions=2,
    obs_dim=5,                    # Dimension of observation vector
    use_fuzzy=True,
    fuzzy_model=fuzzy_expert,     # Pass your pyfume object here
    hyperparameter_update_method="adaptive", # 'adaptive' or 'empirical_bayes'
    lambda_T=1.0,                 # Initial weight for transition prior
    lambda_O=1.0                  # Initial weight for observation prior
)

# Train
log_likelihood = model.fit(observations, actions, max_iterations=100)
```