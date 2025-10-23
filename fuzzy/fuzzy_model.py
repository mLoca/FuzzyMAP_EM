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

def build_fuzzymodel(
    pomdp=None,
    seed: int = 15,
    inputs: list[str] | None = None,
    outputs: list[str] | None = None,
    nr_clus: int = 3,
    trials: int = 215,
    horizon: int = 5,
    save_excel: str | None = None,
):
    """
    Costruisce un modello fuzzy generico.
    - inputs: lista dei nomi delle variabili di input (default: ['test_result','symptoms','action'])
    - outputs: lista dei nomi degli output da modellare (default: ['next_test','next_symptoms'])
    - salva opzionalmente il dataframe in `save_excel`
    Ritorna il modello combinato (pyFUME FIS model).
    """
    np.random.seed(seed)
    random.seed(seed)

    if inputs is None:
        inputs = ['test_result', 'symptoms', 'action']
    if outputs is None:
        outputs = ['next_test', 'next_symptoms']

    # Colleziona dati
    df = collect_data(trials=trials, horizon=horizon, pomdp=pomdp)
    if save_excel:
        df[inputs + outputs].to_excel(save_excel, index=False)

    # Costruisci il FIS base per il primo output con pyFUME
    first_out = outputs[0]
    df_test = df[inputs + [first_out]]
    FIS = pyFUME(dataframe=df_test, nr_clus=nr_clus, variable_names=inputs + [first_out])
    base_model = FIS.FIS.model

    # Assicura che le regole del modello base usino il nome del primo output
    for rule in base_model._rules:
        funstr = rule[1][1]
        rule[1] = (first_out, funstr)

    # Funzione di utilità per ottenere l'indice massimo corrente delle funzioni 'funN'
    def max_fun_index(outputfunctions: dict) -> int:
        nums = [int(k.replace('fun', '')) for k in outputfunctions.keys() if k.startswith('fun')]
        return max(nums) if nums else 0

    # Per ogni output aggiuntivo, stima consequent parameters, costruisci modello e fondi
    for out in outputs[1:]:
        df_test = df[inputs + [out]]
        dl = DataLoader(dataframe=df_test, variable_names=inputs + [out])
        variable_names = dl.variable_names
        ds = DataSplitter()
        x_train, y_train, x_test, y_test = ds.holdout(dataX=dl.dataX, dataY=dl.dataY)

        antecedent_parameters = FIS.FIS.antecedent_parameters
        fsc = FireStrengthCalculator(antecedent_parameters=antecedent_parameters, nr_clus=nr_clus,
                                     variable_names=variable_names)
        firing_strengths = fsc.calculate_fire_strength(data=x_train)

        ce = ConsequentEstimator(x_train=x_train, y_train=y_train, firing_strengths=firing_strengths)
        consequent_parameters = ce.suglms()

        simpbuilder = SugenoFISBuilder(antecedent_sets=antecedent_parameters,
                                       consequent_parameters=consequent_parameters,
                                       variable_names=variable_names)
        new_model = simpbuilder.get_model()

        # Rinomina le output functions del nuovo modello per evitare collisioni
        start_index = max_fun_index(base_model._outputfunctions) + 1
        mapping = {}
        renamed_outputfunctions = {}
        idx = start_index
        for key, val in new_model._outputfunctions.items():
            newkey = f"fun{idx}"
            mapping[key] = newkey
            renamed_outputfunctions[newkey] = val
            idx += 1
        new_model._outputfunctions = renamed_outputfunctions

        # Aggiorna le regole del nuovo modello per usare il nome dell'output corrente e le nuove funzioni
        for rule in new_model._rules:
            old_fun = rule[1][1]
            new_fun = mapping.get(old_fun, old_fun)
            rule[1] = (out, new_fun)

        # Fondi le output functions e le regole nel modello base
        base_model._outputfunctions = {**base_model._outputfunctions, **new_model._outputfunctions}
        base_model._rules = base_model._rules + new_model._rules

    return base_model

def build_fuzzymodel(pomdp = None, seed = 15):
    # Load and learn the fuzzy model from the data with next_test as output variable

    np.random.seed(seed)
    random.seed(seed)

    nr_clus = 3
    df = collect_data(trials=215, horizon=5, pomdp=pomdp)
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