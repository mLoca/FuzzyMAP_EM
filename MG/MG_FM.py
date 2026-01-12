from random import random

from simpful import *
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import operator

STEP_TO_SAVE = [0, 1, 2, 3, 4, 5, 6, 7]  # Steps at which to save the state of the system


def build_fuzzy_model():
    """
    Build a fuzzy model for the immune system dynamics.
    :return: FuzzySystem object with defined variables and rules.
    """
    # Define fuzzy system
    FS = FuzzySystem()
    # Define rule list
    RULES = []
    # Define crisp output values
    FS.set_crisp_output_value('Low', 0)
    FS.set_crisp_output_value('Medium', 0.5)
    FS.set_crisp_output_value('High', 1)

    # Fuzzy rule definition for Teff
    Teff_0 = FuzzySet(points=[[0, 1], [0.2, 1], [0.4, 0], [1, 0]], term='Low')
    Teff_1 = FuzzySet(points=[[0, 0], [0.2, 0], [0.4, 1], [0.6, 1], [0.8, 0], [1, 0]], term='Medium')
    Teff_2 = FuzzySet(points=[[0, 0], [0.6, 0], [0.8, 1], [1, 1]], term='High')
    FS.add_linguistic_variable('Teff', LinguisticVariable([Teff_0, Teff_1, Teff_2], concept='Teff'))
    # Fuzzy rule definition for Treg
    Treg_0 = FuzzySet(points=[[0, 1], [0.2, 1], [0.4, 0], [1, 0]], term='Low')
    Treg_1 = FuzzySet(points=[[0, 0], [0.2, 0], [0.4, 1], [0.6, 1], [0.8, 0], [1, 0]], term='Medium')
    Treg_2 = FuzzySet(points=[[0, 0], [0.6, 0], [0.8, 1], [1, 1]], term='High')
    FS.add_linguistic_variable('Treg', LinguisticVariable([Treg_0, Treg_1, Treg_2], concept='Treg'))
    # Fuzzy rule definition for B
    B_0 = FuzzySet(points=[[0, 1], [0.2, 1], [0.4, 0], [1, 0]], term='Low')
    B_1 = FuzzySet(points=[[0, 0], [0.2, 0], [0.4, 1], [0.6, 1], [0.8, 0], [1, 0]], term='Medium')
    B_2 = FuzzySet(points=[[0, 0], [0.6, 0], [0.8, 1], [1, 1]], term='High')
    FS.add_linguistic_variable('B', LinguisticVariable([B_0, B_1, B_2], concept='B'))
    # Fuzzy rule definition for GC
    GC_0 = FuzzySet(points=[[0, 1], [0.2, 1], [0.4, 0], [1, 0]], term='Low')
    GC_1 = FuzzySet(points=[[0, 0], [0.2, 0], [0.4, 1], [0.6, 1], [0.8, 0], [1, 0]], term='Medium')
    GC_2 = FuzzySet(points=[[0, 0], [0.6, 0], [0.8, 1], [1, 1]], term='High')
    FS.add_linguistic_variable('GC', LinguisticVariable([GC_0, GC_1, GC_2], concept='GC'))
    # Fuzzy rule definition for SLPB
    SLPB_0 = FuzzySet(points=[[0, 1], [0.2, 1], [0.4, 0], [1, 0]], term='Low')
    SLPB_1 = FuzzySet(points=[[0, 0], [0.2, 0], [0.4, 1], [0.6, 1], [0.8, 0], [1, 0]], term='Medium')
    SLPB_2 = FuzzySet(points=[[0, 0], [0.6, 0], [0.8, 1], [1, 1]], term='High')
    FS.add_linguistic_variable('SLPB', LinguisticVariable([SLPB_0, SLPB_1, SLPB_2], concept='SLPB'))
    # Fuzzy rule definition for LLPC
    LLPC_0 = FuzzySet(points=[[0, 1], [0.2, 1], [0.4, 0], [1, 0]], term='Low')
    LLPC_1 = FuzzySet(points=[[0, 0], [0.2, 0], [0.4, 1], [0.6, 1], [0.8, 0], [1, 0]], term='Medium')
    LLPC_2 = FuzzySet(points=[[0, 0], [0.6, 0], [0.8, 1], [1, 1]], term='High')
    FS.add_linguistic_variable('LLPC', LinguisticVariable([LLPC_0, LLPC_1, LLPC_2], concept='LLPC'))
    # Fuzzy rule definition for IgG
    IgG_0 = FuzzySet(points=[[0, 1], [0.2, 1], [0.4, 0], [1, 0]], term='Low')
    IgG_1 = FuzzySet(points=[[0, 0], [0.2, 0], [0.4, 1], [0.6, 1], [0.8, 0], [1, 0]], term='Medium')
    IgG_2 = FuzzySet(points=[[0, 0], [0.6, 0], [0.8, 1], [1, 1]], term='High')
    FS.add_linguistic_variable('IgG', LinguisticVariable([IgG_0, IgG_1, IgG_2], concept='IgG'))
    # Fuzzy rule definition for Complement
    Complement_0 = FuzzySet(points=[[0, 1], [0.2, 1], [0.4, 0], [1, 0]], term='Low')
    Complement_1 = FuzzySet(points=[[0, 0], [0.2, 0], [0.4, 1], [0.6, 1], [0.8, 0], [1, 0]], term='Medium')
    Complement_2 = FuzzySet(points=[[0, 0], [0.6, 0], [0.8, 1], [1, 1]], term='High')
    FS.add_linguistic_variable('Complement',
                               LinguisticVariable([Complement_0, Complement_1, Complement_2], concept='Complement'))
    # Fuzzy rule definition for Symptoms
    Symptoms_0 = FuzzySet(points=[[0, 1], [0.2, 1], [0.4, 0], [1, 0]], term='Low')
    Symptoms_1 = FuzzySet(points=[[0, 0], [0.2, 0], [0.4, 1], [0.6, 1], [0.8, 0], [1, 0]], term='Medium')
    Symptoms_2 = FuzzySet(points=[[0, 0], [0.6, 0], [0.8, 1], [1, 1]], term='High')
    FS.add_linguistic_variable('Symptoms', LinguisticVariable([Symptoms_0, Symptoms_1, Symptoms_2], concept='Symptoms'))
    # Fuzzy rule definition for Inflammation
    Inflammation_0 = FuzzySet(points=[[0, 1], [0.2, 1], [0.4, 0], [1, 0]], term='Low')
    Inflammation_1 = FuzzySet(points=[[0, 0], [0.2, 0], [0.4, 1], [0.6, 1], [0.8, 0], [1, 0]], term='Medium')
    Inflammation_2 = FuzzySet(points=[[0, 0], [0.6, 0], [0.8, 1], [1, 1]], term='High')
    FS.add_linguistic_variable('Inflammation', LinguisticVariable([Inflammation_0, Inflammation_1, Inflammation_2],
                                                                  concept='Inflammation'))

    # Fuzzy set definition for Ravu
    Ravu_0 = FuzzySet(points=[[0, 1], [0.2, 1], [0.8, 0], [1, 0]], term='Low')
    Ravu_1 = FuzzySet(points=[[0, 0], [0.2, 0], [0.8, 1], [1, 1]], term='High')
    FS.add_linguistic_variable('Ravu', LinguisticVariable([Ravu_0, Ravu_1], concept='Ravu'))
    #FS.plot_variable('Ravu')

    # Define rules for Ravu
    RULES.append('IF (Ravu IS High) THEN (Complement IS Low)')

    # Define rules for Teff
    RULES.append('IF (Inflammation IS High) THEN (Teff IS High)')
    RULES.append('IF (Inflammation IS Medium) THEN (Teff IS Medium)')
    RULES.append('IF (Inflammation IS Low) THEN (Teff IS Low)')
    RULES.append('IF (Treg IS High) THEN (Teff IS Low)')
    RULES.append('IF (Treg IS Medium) THEN (Teff IS Medium)')
    RULES.append('IF (Treg IS Low) THEN (Teff IS High)')

    # Define rules for Treg
    RULES.append('IF (Inflammation IS High) THEN (Treg IS Low)')
    RULES.append('IF (Inflammation IS Medium) THEN (Treg IS Medium)')
    RULES.append('IF (Inflammation IS Low) THEN (Treg IS High)')

    # Define rules for B
    RULES.append('IF (Teff IS High) THEN (B IS High)')
    RULES.append('IF (Teff IS Medium) THEN (B IS Medium)')
    RULES.append('IF (Teff IS Low) THEN (B IS Low)')
    RULES.append('IF (Inflammation IS High) THEN (B IS High)')
    RULES.append('IF (Inflammation IS Medium) THEN (B IS Medium)')
    RULES.append('IF (Inflammation IS Low) THEN (B IS Low)')
    RULES.append('IF (Treg IS High) THEN (B IS Low)')
    RULES.append('IF (Treg IS Medium) THEN (B IS Medium)')
    RULES.append('IF (Treg IS Low) THEN (B IS High)')

    # Define rules for GC
    RULES.append('IF (Teff IS High) AND (B IS High) AND (Complement IS High) THEN (GC IS High)')
    RULES.append('IF (Teff IS High) AND (B IS High) AND (Complement IS Medium) THEN (GC IS Medium)')
    RULES.append('IF (Teff IS High) AND (B IS Medium) AND (Complement IS High) THEN (GC IS Medium)')
    RULES.append('IF (Teff IS High) AND (B IS Medium) AND (Complement IS Medium) THEN (GC IS Medium)')
    RULES.append('IF (Teff IS Medium) AND (B IS High) AND (Complement IS High) THEN (GC IS High)')
    RULES.append('IF (Teff IS Medium) AND (B IS High) AND (Complement IS Medium) THEN (GC IS Medium)')
    RULES.append('IF (Teff IS Medium) AND (B IS Medium) AND (Complement IS High) THEN (GC IS Medium)')
    RULES.append('IF (Teff IS Medium) AND (B IS Medium) AND (Complement IS Medium) THEN (GC IS Medium)')
    RULES.append('IF (Teff IS Low) THEN (GC IS Low)')
    RULES.append('IF (B IS Low) THEN (GC IS Low)')
    RULES.append('IF (Complement IS Low) THEN (GC IS Low)')
    RULES.append('IF (Treg IS High) THEN (GC IS Low)')
    RULES.append('IF (Treg IS Medium) THEN (GC IS Medium)')
    RULES.append('IF (Treg IS Low) THEN (GC IS High)')

    # Define rules for SLPB
    RULES.append('IF (B IS High) THEN (SLPB IS High)')
    RULES.append('IF (B IS Medium) THEN (SLPB IS Medium)')
    RULES.append('IF (B IS Low) THEN (SLPB IS Low)')

    # Define rules for LLPC
    RULES.append('IF (GC IS High) OR (LLPC IS High) THEN (LLPC IS High)')
    RULES.append('IF (GC IS Medium) OR (LLPC IS Medium) THEN (LLPC IS Medium)')
    RULES.append('IF (GC IS Low) AND (LLPC IS Low) THEN (LLPC IS Low)')

    # Define rules for IgG
    RULES.append('IF (SLPB IS High) THEN (IgG IS High)')
    RULES.append('IF (SLPB IS Medium) THEN (IgG IS Medium)')
    RULES.append('IF (SLPB IS Low) THEN (IgG IS Low)')
    RULES.append('IF (LLPC IS High) THEN (IgG IS High)')
    RULES.append('IF (LLPC IS Medium) THEN (IgG IS Medium)')
    RULES.append('IF (LLPC IS Low) THEN (IgG IS Low)')

    # Define rules for Complement
    RULES.append('IF (IgG IS High) THEN (Complement IS High)')
    RULES.append('IF (IgG IS Medium) THEN (Complement IS Medium)')
    RULES.append('IF (IgG IS Low) THEN (Complement IS Low)')

    # Define rules for Symptoms
    RULES.append('IF (IgG IS High) THEN (Symptoms IS High)')
    RULES.append('IF (IgG IS Medium) THEN (Symptoms IS Medium)')
    RULES.append('IF (IgG IS Low) THEN (Symptoms IS Low)')
    RULES.append('IF (Complement IS High) THEN (Symptoms IS High)')
    RULES.append('IF (Complement IS Medium) THEN (Symptoms IS Medium)')
    RULES.append('IF (Complement IS Low) THEN (Symptoms IS Low)')

    # add fuzzy rules
    FS.add_rules(RULES)

    return FS


def _simulate_data(FS_model, trials=1, length=7):
    """
    Simulate data for the fuzzy system.
    :param FS: FuzzySystem object.
    :return: Dictionary with variable names and their values.
    """
    observations = []
    actions = []
    steps = 14  # Total steps to simulate
    for trail in range(trials):
        init_state = {
            'Teff': np.random.random(),
            'Treg': np.random.random(),
            'B': np.random.random(),
            'GC': np.random.random(),
            'SLPB': np.random.random(),
            'LLPC': np.random.random(),
            'IgG': np.random.random(),
            'Complement': np.random.random(),
            'Symptoms': np.random.random(),
            'Inflammation': np.random.random()
        }  # Save initial state
        #for treatment in [0, 1]:

        treatment = trail % 2  # Alternate treatment between 0 and 1

        FS = _initialize_new_patient(FS_model, init_state)
        FS.set_variable('Ravu', treatment)  # Set RAVU active

        observation = np.zeros((length, len(FS._variables) - 1))
        if treatment == 0:
            action = np.zeros((length, 1), dtype=int)  # Initialize action array
        else:
            action = np.ones((length, 1), dtype=int)  # Initialize action array
        var_names = list(FS._variables.keys())

        # Memorizza i valori iniziali
        #for i, var in enumerate(var_names):
        #    if var != 'Ravu':
        #        observation[0, i] = FS._variables[var]

        dynamics = deepcopy(FS._variables)
        for var in dynamics.keys():
            dynamics[var] = [dynamics[var]]
        # Perform Sugeno inference and save results
        step = 0
        for T in np.linspace(0, 1, steps):
            FS.set_variable('Ravu', treatment)

            #choose the treatment action for the current step randomly
            #if step >= 0 and step in STEP_TO_SAVE:
            #   treatment = 0 if random() < 0.5 else 1
            #   FS.set_variable('Ravu', treatment)
            #   index_step = STEP_TO_SAVE.index(step)
            #   action[index_step] = treatment
            #f step > 0 and step in STEP_TO_SAVE:
            #   treatment = np.abs(trail % 2 - 1)
            #   if step > STEP_TO_SAVE[-1] / 2:
            #       FS.set_variable('Ravu', treatment)
            #   else:
            #       FS.set_variable('Ravu', np.abs(treatment - 1))
            #   index_step = STEP_TO_SAVE.index(step)
            #   action[index_step] = FS._variables['Ravu']

            new_values = FS.Sugeno_inference()

            for var in new_values.keys():
                dynamics[var].append(new_values[var])
            FS._variables.update(new_values)
            # Perturbations can be added here using the set_variable method
            if step in STEP_TO_SAVE:
                index_var = 0
                index_step = STEP_TO_SAVE.index(step)
                for var in FS._variables.keys():
                    if var != 'Ravu':
                        if var == 'Inflammation':
                            observation[index_step, index_var] = FS._variables[var]
                        else:
                            observation[index_step, index_var] = new_values[var]
                        index_var += 1

            step += 1

        observations.append(observation)
        actions.append(action)
    return observations, actions


def _initialize_new_patient(FS, patient_data):
    """
    Initialize a new patient with given data.
    :param FS: FuzzySystem object.
    :param patient_data: Dictionary with patient data.
    """
    for var, val in patient_data.items():
        FS.set_variable(var, val)

    return FS


def symptoms_with_treatment(variable="Symptoms"):
    """
    Simulate symptoms dynamics with treatment.
    :return: None
    """
    seed = 55
    trails = range(50)
    np.random.seed(seed)

    FS = build_fuzzy_model()
    steps = 9
    total_symptoms = [np.zeros(10), np.zeros(10)]
    counts = [0, 0]
    for t in trails:
        init_state = {
            'Teff': np.random.random(),
            'Treg': np.random.random(),
            'B': np.random.random(),
            'GC': np.random.random(),
            'SLPB': np.random.random(),
            'LLPC': np.random.random(),
            'IgG': np.random.random(),
            'Complement': np.random.random(),
            'Symptoms': np.random.random(),
            'Inflammation': np.random.random()
        }
        for treatment in [0, 1]:
            FS = _initialize_new_patient(FS, init_state)
            FS.set_variable('Ravu', treatment)  # Set RAVU active/inactive
            dynamics = deepcopy(FS._variables)
            for var in dynamics.keys():
                dynamics[var] = [dynamics[var]]

            for T in np.linspace(0, 1, steps):
                new_values = FS.Sugeno_inference()
                FS._variables.update(new_values)
                for var in new_values.keys():
                    noise = np.random.normal(0, 0.03)
                    clipped_value = np.clip(new_values[var] + noise, 0, 1)
                    new_values[var] = clipped_value
                    dynamics[var].append(clipped_value)
                noise = np.random.normal(0, 0.03)
                new_values_inf = dynamics["Inflammation"][-1]
                clipped_value = np.clip(new_values_inf + noise, 0, 1)
                dynamics["Inflammation"].append(clipped_value)

            Symptoms = dynamics[variable]

            total_symptoms[treatment] = list(map(operator.add, total_symptoms[treatment], Symptoms))
            counts[treatment] += 1

    # Calcola la media dei sintomi per ogni passo temporale
    total_symptoms[0] = [x / counts[0] for x in total_symptoms[0]]
    total_symptoms[1] = [x / counts[1] for x in total_symptoms[1]]

    plt.plot(range(steps + 1), total_symptoms[0], label=f'Treatment={0}')
    plt.plot(range(steps + 1), total_symptoms[1], label=f'Treatment={1}')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    #symptoms_with_treatment("B")
    FS = build_fuzzy_model()

    for trial in range(25):

        init_state = {
            'Teff': np.random.random(),
            'Treg': np.random.random(),
            'B': np.random.random(),
            'GC': np.random.random(),
            'SLPB': np.random.random(),
            'LLPC': np.random.random(),
            'IgG': np.random.random(),
            'Complement': np.random.random(),
            'Symptoms': np.random.random(),
            'Inflammation': np.random.random()
        }
        for t in range(2):
            FS = _initialize_new_patient(FS, init_state)

            FS.set_variable('Ravu', t)  # Set RAVU inactive
            # Set number of inference steps and save initial state
            steps = 30
            dynamics = deepcopy(FS._variables)
            for var in dynamics.keys():
                dynamics[var] = [dynamics[var]]
            # Perform Sugeno inference and save results
            for T in np.linspace(0, 1, steps):
                FS.set_variable('Ravu', 0)
                if T > 0.5:
                    FS.set_variable('Ravu', 1)
                new_values = FS.Sugeno_inference()
                FS._variables.update(new_values)
                # Perturbations can be added here using the set_variable method
                for var in new_values.keys():
                    dynamics[var].append(new_values[var])

            #Plotting dynamics
            Teff = dynamics['Teff']
            Treg = dynamics['Treg']
            B = dynamics['B']
            IgG = dynamics['IgG']
            LLPC = dynamics['LLPC']
            Complement = dynamics['Complement']
            Symptoms = dynamics['Symptoms']
            plt.plot(range(steps + 1), Teff)
            plt.plot(range(steps + 1), Treg)
            plt.plot(range(steps + 1), B)
            plt.plot(range(steps + 1), IgG)
            plt.plot(range(steps + 1), LLPC)
            plt.plot(range(steps + 1), Complement)
            plt.plot(range(steps + 1), Symptoms)
            plt.ylim(0, 1.05)
            plt.xlabel('Time')
            plt.ylabel('Level')
            plt.legend(['Teff', 'Treg', 'B', 'IgG', 'LLPC', 'Complement', 'Symptoms'], loc='lower right',
                       framealpha=0.9)
            plt.show()
