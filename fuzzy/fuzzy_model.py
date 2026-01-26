import random
import numpy as np
import pandas as pd
from typing import List, Optional, Union, Dict, Any
from sklearn.metrics import mean_squared_error
from pyfume import pyFUME, DataSplitter, DataLoader, FireStrengthCalculator, ConsequentEstimator, SugenoFISBuilder

from envs.continuous_medical_pomdp import create_continuous_medical_pomdp, ACTIONS, reset_pomdp


def collect_data(
        trials: int = 1000,
        horizon: int = 3,
        noise: float = 0.0,
        pomdp=None,
        target_per_state: int = 130,
        state_in_df=False
):
    """
    Collects interaction data from the POMDP environment.
    Uses a balancing strategy to ensure enough coverage of critical/sick states.
    """
    data = []
    count = {'critical': 0, 'sick': 0, 'healthy': 0}
    episodes_run = 0

    while episodes_run < trials:
        # prepare/ reset pomdp
        if pomdp is None:
            pomdp = create_continuous_medical_pomdp()
        else:
            pomdp = reset_pomdp(pomdp)

        observation = pomdp.agent.observation_model.sample(pomdp.env.state, None)
        count[pomdp.env.state.name] += 1

        total_reward = 0
        for step in range(horizon - 1):
            choices = []
            # deficits to target specific states
            deficits = {s: max(0, target_per_state - c) for s, c in count.items()}
            for s, d in deficits.items():
                if d > 0:
                    choices.extend([s] * d)

            action_idx = random.randint(0, len(ACTIONS) - 1)
            action = ACTIONS[action_idx]
            applied_transition = False

            #force a transition to a needed state if under-represented
            next_state = None
            if len(choices) == 0:
                reward = pomdp.env.state_transition(action, execute=True)
                next_state = pomdp.env.state
            else:
                for _ in range(len(choices)):
                    cand_next_state, cand_reward = pomdp.env.state_transition(action, execute=False)
                    if cand_next_state.name in choices:
                        pomdp.env.apply_transition(cand_next_state)
                        next_state = pomdp.env.state
                        reward = cand_reward
                        applied_transition = True
                        break

            if not applied_transition:
                # fallback to performing the real transition
                reward = pomdp.env.state_transition(action, execute=True)
                next_state = pomdp.env.state

            next_observation = pomdp.agent.observation_model.sample(next_state, action)
            next_observation = next_observation + np.random.normal(0, noise, size=len(next_observation))
            next_observation = np.clip(next_observation, 0.0, 1.0)

            count[next_state.name] += 1

            trajectory_step = {
                'test': observation[0],
                'symptoms': observation[1],
                'action': action_idx,
                'reward': reward,
                'next_test': next_observation[0],
                'next_symptoms': next_observation[1]
            }
            if state_in_df:
                trajectory_step['state'] = pomdp.env.state.name

            data.append(trajectory_step)
            total_reward += reward
            observation = next_observation

        episodes_run += 1

    print(f"Total critical states encountered: {count['critical']}")
    print(f"Total sick states encountered: {count['sick']}")
    print(f"Total healthy states encountered: {count['healthy']}")

    df = pd.DataFrame(data)
    return df

def build_fuzzymodel(pomdp=None, seed=42, trails=300, noise=0):
    # Load and learn the fuzzy model from the data with next_test as output variable
    np.random.seed(seed)
    random.seed(seed)

    nr_clus = 5
    df = collect_data(trials=trails, horizon=3, pomdp=pomdp, target_per_state=220, noise=noise, state_in_df=False)
    df["action"] = df["action"].astype("category")

    # df to excel
    df_test = df[["test", "symptoms", "action", "next_test"]]
    FIS = pyFUME(dataframe=df_test, nr_clus=nr_clus,
                 variable_names=['test', 'symptoms', 'action', 'next_test'],
                 verbose=False)
    model = []

    #Add the rules of the FIS model to include the next_symptoms as output variable
    for out in ["next_symptoms"]:
        df_test = df[["test", "symptoms", "action", out]]
        dl = DataLoader(dataframe=df_test,
                        variable_names=['test', 'symptoms', 'action', out], verbose=False)
        variable_names = dl.variable_names

        # Split the data using the hold-out method in a training (default: 75%)
        ds = DataSplitter()
        x_train, y_train, x_test, y_test = ds.holdout(dataX=dl.dataX, dataY=dl.dataY)

        # Take the memberships of the antecedent sets
        antecedent_parameters = FIS.FIS.antecedent_parameters

        # Cluster the antecedent parameters to reduce the number of rules
        fsc = FireStrengthCalculator(antecedent_parameters=antecedent_parameters, nr_clus=nr_clus,
                                     variable_names=variable_names, verbose=False)
        firing_strengths = fsc.calculate_fire_strength(data=x_train)

        # Estimate the parameters of the consequent functions
        ce = ConsequentEstimator(x_train=x_train, y_train=y_train, firing_strengths=firing_strengths)
        consequent_parameters = ce.suglms()

        # Build a first-order Takagi-Sugeno model.
        simpbuilder = SugenoFISBuilder(antecedent_sets=antecedent_parameters,
                                       consequent_parameters=consequent_parameters,
                                       variable_names=variable_names,
                                       verbose=False)
        model.append(simpbuilder.get_model())

        # Change the name of the functions in the model to avoid conflicts
        count = len(FIS.FIS.model._outputfunctions) + 1
        new_dict = {}
        for key, val in model[0]._outputfunctions.items():
            newkey = "fun" + str(count)
            count += 1
            new_dict[newkey] = val
        model[0]._outputfunctions = new_dict

        # Change the rules in the model to include the next_symptoms
        for rule in FIS.FIS.model._rules:
            funstr = rule[1][1]
            rule[1] = ('next_test', funstr)

        for rule in model[0]._rules:
            funstr = rule[1][1]
            number_fun = int(funstr.split("fun")[1])
            new_number = number_fun + len(FIS.FIS.model._outputfunctions)
            new_funstr = "fun" + str(new_number)
            rule[1] = ('next_symptoms', new_funstr)


        FIS.FIS.model._outputfunctions = FIS.FIS.model._outputfunctions | model[0]._outputfunctions
        FIS.FIS.model._rules = FIS.FIS.model._rules + model[0]._rules

        return FIS.FIS.model


def _predict(model, input_data):
    for key, value in input_data.items():
        model.set_variable(key, value)
    pred_obs = model.Sugeno_inference(["next_test", "next_symptoms"])
    return pred_obs


def _evaluate_model(model, df, inputs=None, outputs=None):
    if inputs is None:
        inputs = ['test', 'symptoms', 'action']
    if outputs is None:
        outputs = ['next_test', 'next_symptoms']

    predictions = {out: [] for out in outputs}
    real_observations = {out: df[out].values for out in outputs}
    states = df['state'].values

    # Iterate row by row to predict
    for _, row in df.iterrows():
        input_dict = {inp: row[inp] for inp in inputs}
        res = _predict(model, input_dict)
        for out in outputs:
            predictions[out].append(res.get(out, 0.0))

    for out in outputs:
        predictions[out] = np.array(predictions[out])

    print("\n--- Global MSE ---")
    for out in outputs:
        r2 = mean_squared_error(real_observations[out], predictions[out])
        print(f"Global MSE for '{out}': {r2:.4f}")

    print("\n--- State-by-State Performance ---")
    unique_states = np.unique(states)
    for state in unique_states:
        idx = (states == state)
        if not any(idx): continue

        print(f"State: {state.upper()}")
        for out in outputs:
            y_true_s = real_observations[out][idx]
            y_pred_s = predictions[out][idx]

            r2_s = mean_squared_error(y_true_s, y_pred_s)
            print(f"  MSE for '{out}': {r2_s:.4f}")


if __name__ == "__main__":
    build_fuzzymodel()
