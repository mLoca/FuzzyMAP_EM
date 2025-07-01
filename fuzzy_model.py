import random
from typing import Tuple

import numpy as np
from continouos_pomdp_example import create_continuous_medical_pomdp, ACTIONS, reset_pomdp
import pandas as pd
from pyfume import BuildTakagiSugeno, pyFUME, SugenoFISTester, DataSplitter, FeatureSelector, DataLoader, Clusterer, \
    AntecedentEstimator, FireStrengthCalculator, ConsequentEstimator, SugenoFISBuilder




def collect_data(trials=1000, horizon=10, pomdp=None):
    data = []

    count ={
        'critical': 0,
        'sick': 0,
        'healthy': 0,
    }

    for episode in range(trials):
        #random.seed(episode)  # For reproducibility
        # Reset environment with random initial state
        if pomdp is None:
            pomdp = create_continuous_medical_pomdp()
        else:
            pomdp = reset_pomdp(pomdp)

        observation = pomdp.agent.observation_model.sample(pomdp.env.state, None)
        total_reward = 0
        if pomdp.env.state.name == "critical":
            count["critical"] += 1
        if pomdp.env.state.name == "sick":
            count["sick"] += 1
        if pomdp.env.state.name == "healthy":
            count["healthy"] += 1

        for step in range(horizon):
            # Random action
            action_idx = random.randint(0, len(ACTIONS) - 1)
            action = ACTIONS[action_idx]

            # Execute action
            reward = pomdp.env.state_transition(action, execute=True)
            next_state = pomdp.env.state
            next_observation = pomdp.agent.observation_model.sample(next_state, action)

            if next_state.name == "critical":
                count["critical"] += 1
            if next_state.name == "sick":
                count["sick"] += 1
            if next_state.name == "healthy":
                count["healthy"] += 1

            # Collect data
            data.append({
                'test_result': observation[0],
                'symptoms': observation[1],
                'action': action_idx,
                'reward': reward,
                'next_test': next_observation[0],
                'next_symptoms': next_observation[1]
            })

            total_reward += reward
            observation = next_observation


    print(f"Total critical states encountered: {count['critical']}")
    print(f"Total sick states encountered: {count['sick']}")
    print(f"Total healthy states encountered: {count['healthy']}")
    # Create DataFrame from collected data
    df = pd.DataFrame(data)
    return df


def build_fuzzymodel(pomdp = None):
    # Load and learn the fuzzy model from the data with next_test as output variable

    np.random.seed(15)
    random.seed(15)

    nr_clus = 3
    df = collect_data(trials=600, horizon=10, pomdp=pomdp)
    # df to excel
    df[["test_result", "symptoms", "action", "next_test", "next_symptoms"]].to_excel("data.xlsx", index=False)
    df_test = df[["test_result", "symptoms", "action", "next_test"]]
    FIS = pyFUME(dataframe=df_test, nr_clus=nr_clus,
                 variable_names=['test_result', 'symptoms', 'action', 'next_test'], )

    #print("the mean squared error of the created model is", FIS.calculate_error("MSE"))
    #print("the mean area error of the created model is", FIS.calculate_error("MAE"))
    model = []

    #Add the rules of the FIS model to include the next_symptoms as output variable
    for out in ["next_symptoms"]:
        df_test = df[["test_result", "symptoms", "action", out]]
        dl = DataLoader(dataframe=df_test,
                        variable_names=['test_result', 'symptoms', 'action', out], )
        variable_names = dl.variable_names
        dataX = dl.dataX
        dataY = dl.dataY

        # Split the data using the hold-out method in a training (default: 75%)
        # and test set (default: 25%).
        ds = DataSplitter()
        x_train, y_train, x_test, y_test = ds.holdout(dataX=dl.dataX, dataY=dl.dataY)

        # Take the memberships of the antecedent sets
        antecedent_parameters = FIS.FIS.antecedent_parameters

        # Cluster the antecedent parameters to reduce the number of rules
        fsc = FireStrengthCalculator(antecedent_parameters=antecedent_parameters, nr_clus=nr_clus,
                                     variable_names=variable_names)
        firing_strengths = fsc.calculate_fire_strength(data=x_train)

        # Estimate the parameters of the consequent functions
        ce = ConsequentEstimator(x_train=x_train, y_train=y_train, firing_strengths=firing_strengths)
        consequent_parameters = ce.suglms()

        # Build a first-order Takagi-Sugeno model.
        simpbuilder = SugenoFISBuilder(antecedent_sets=antecedent_parameters, consequent_parameters=consequent_parameters,
                                       variable_names=variable_names)
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

        # Plot the model

        FIS.FIS.model._outputfunctions = FIS.FIS.model._outputfunctions | model[0]._outputfunctions
        FIS.FIS.model._rules = FIS.FIS.model._rules + model[0]._rules


        return FIS.FIS.model




if __name__ == "__main__":
    build_fuzzymodel()