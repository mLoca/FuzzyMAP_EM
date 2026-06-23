from simpful import *
import numpy as np
import matplotlib.pyplot as plt
import copy

class HIVExpertV2Model:
    def __init__(self):
        """
        Modified HIV fuzzy system from v2.
        Scaled to [-5, 5] (standardized log space) to be compatible with the HIV example POMDP observations.
        """
        self.FS = FuzzySystem(show_banner=False)
        RULES = []
        
        # Define crisp output values (scaled from [0, 1000] -> [-5, 5])
        self.FS.set_crisp_output_value('Depleted', -5)
        self.FS.set_crisp_output_value('Partially_depleted', -1)
        self.FS.set_crisp_output_value('Abundant', 5)
        self.FS.set_crisp_output_value('Low', -5)
        self.FS.set_crisp_output_value('High', 5)

        # Treatment Action (0=None, 1=RTI, 2=PI, 3=Both)
        A_none = FuzzySet(function=Triangular_MF(a=0.0, b=0.0, c=0.25), term="None")
        A_rti = FuzzySet(function=Triangular_MF(a=0.75, b=1.0, c=1.25), term="RTI")
        A_pi = FuzzySet(function=Triangular_MF(a=1.75, b=2.0, c=2.25), term="PI")
        A_both = FuzzySet(function=Triangular_MF(a=2.75, b=3.0, c=3.25), term="Both")
        self.FS.add_linguistic_variable("Action", LinguisticVariable([A_none, A_rti, A_pi, A_both], universe_of_discourse=[0, 3]))

        # T1 (combines old T1 and T1p sets)
        T1_0 = FuzzySet(points=[[-5.0, 1], [-4.0, 1], [-3.0, 0], [5.0, 0]], term='Depleted')
        T1_1 = FuzzySet(points=[[-5.0, 0], [-4.0, 0], [-3.0, 1], [-1.0, 1], [0.0, 0], [5.0, 0]], term='Partially_depleted')
        T1_2 = FuzzySet(points=[[-5.0, 0], [-1.0, 0], [0.0, 1], [5.0, 1]], term='Abundant')
        T1_3 = FuzzySet(points=[[-5.0, 1], [-4.9, 1], [4.9, 0], [5.0, 0]], term='Low')
        T1_4 = FuzzySet(points=[[-5.0, 0], [-4.9, 0], [4.9, 1], [5.0, 1]], term='High')
        self.FS.add_linguistic_variable('T1', LinguisticVariable([T1_0, T1_1, T1_2, T1_3, T1_4], concept='T1', universe_of_discourse=[-5, 5]))

        # T1_inf
        T1_inf_0 = FuzzySet(points=[[-5.0, 1], [-4.0, 1], [-3.0, 0], [5.0, 0]], term='Depleted')
        T1_inf_1 = FuzzySet(points=[[-5.0, 0], [-4.0, 0], [-3.0, 1], [-1.0, 1], [0.0, 0], [5.0, 0]], term='Partially_depleted')
        T1_inf_2 = FuzzySet(points=[[-5.0, 0], [-1.0, 0], [0.0, 1], [5.0, 1]], term='Abundant')
        self.FS.add_linguistic_variable('T1_inf', LinguisticVariable([T1_inf_0, T1_inf_1, T1_inf_2], concept='T1_inf', universe_of_discourse=[-5, 5]))

        # T2 (combines old T2 and T2p sets)
        T2_0 = FuzzySet(points=[[-5.0, 1], [-4.9, 1], [4.9, 0], [5.0, 0]], term='Low')
        T2_1 = FuzzySet(points=[[-5.0, 0], [-4.9, 0], [4.9, 1], [5.0, 1]], term='High')
        self.FS.add_linguistic_variable('T2', LinguisticVariable([T2_0, T2_1], concept='T2', universe_of_discourse=[-5, 5]))

        # T2_inf
        T2_inf_0 = FuzzySet(points=[[-5.0, 1], [-4.9, 1], [4.9, 0], [5.0, 0]], term='Low')
        T2_inf_1 = FuzzySet(points=[[-5.0, 0], [-4.9, 0], [4.9, 1], [5.0, 1]], term='High')
        self.FS.add_linguistic_variable('T2_inf', LinguisticVariable([T2_inf_0, T2_inf_1], concept='T2_inf', universe_of_discourse=[-5, 5]))

        # V
        V_0 = FuzzySet(points=[[-5.0, 1], [-4.9, 1], [4.9, 0], [5.0, 0]], term='Low')
        V_1 = FuzzySet(points=[[-5.0, 0], [-4.9, 0], [4.9, 1], [5.0, 1]], term='High')
        self.FS.add_linguistic_variable('V', LinguisticVariable([V_0, V_1], concept='V', universe_of_discourse=[-5, 5]))

        # E
        E_0 = FuzzySet(points=[[-5.0, 1], [-4.9, 1], [4.9, 0], [5.0, 0]], term='Low')
        E_1 = FuzzySet(points=[[-5.0, 0], [-4.9, 0], [4.9, 1], [5.0, 1]], term='High')
        self.FS.add_linguistic_variable('E', LinguisticVariable([E_0, E_1], concept='E', universe_of_discourse=[-5, 5]))
        self.FS.plot_variable('E')

        # Define rules for T1
        RULES.append('IF (T1 IS High) THEN (T1 IS Abundant)')
        RULES.append('IF (T1 IS Low) THEN (T1 IS Depleted)')
        RULES.append('IF (V IS High) THEN (T1 IS Partially_depleted)')

        # Define rules for T1_inf
        RULES.append('IF (V IS High) AND (T1 IS Abundant) THEN (T1_inf IS Abundant)')
        RULES.append('IF (V IS High) AND (T1 IS Partially_depleted) THEN (T1_inf IS Partially_depleted)')
        RULES.append('IF (V IS High) AND (T1 IS Depleted) THEN (T1_inf IS Depleted)')
        RULES.append('IF (V IS Low) THEN (T1_inf IS Depleted)')
        RULES.append('IF (E IS High) THEN (T1_inf IS Depleted)')
        RULES.append('IF (Action IS RTI) THEN (T1_inf IS Depleted)')
        RULES.append('IF (Action IS Both) THEN (T1_inf IS Depleted)')

        # Define rules for T2
        RULES.append('IF (T2 IS High) THEN (T2 IS High)')
        RULES.append('IF (T2 IS Low) THEN (T2 IS Low)')
        RULES.append('IF (V IS High) THEN (T2 IS Low)')

        # Define rules for T2_inf
        RULES.append('IF (V IS High) AND (T2 IS High) THEN (T2_inf IS High)')
        RULES.append('IF (V IS High) AND (T2 IS Low) THEN (T2_inf IS Low)')
        RULES.append('IF (V IS Low) THEN (T2_inf IS Low)')
        RULES.append('IF (E IS High) THEN (T2_inf IS Low)')
        RULES.append('IF (Action IS RTI) THEN (T2_inf IS Low)')
        RULES.append('IF (Action IS Both) THEN (T2_inf IS Low)')

        # Define rules for V
        RULES.append('IF (T1_inf IS Abundant) OR (T1_inf IS Partially_depleted) OR (T2_inf IS High) THEN (V IS High)')
        RULES.append('IF (T1_inf IS Depleted) AND (T2_inf IS Low) THEN (V IS Low)')
        RULES.append('IF (Action IS PI) THEN (V IS Low)')
        RULES.append('IF (Action IS Both) THEN (V IS Low)')

        # Define rules for E
        RULES.append('IF (T1_inf IS Abundant) THEN (E IS Low)')
        RULES.append('IF (T1_inf IS Partially_depleted) THEN (E IS High)')
        RULES.append('IF (T1_inf IS Depleted) THEN (E IS Low)')
        RULES.append('IF (V IS High) THEN (E IS Low)')
        RULES.append('IF (V IS Low) THEN (E IS High)')

        self.FS.add_rules(RULES)

    def get_model(self):
        return self.FS

def build_fuzzy_model():
    """
    Function interface expected by some scripts to easily retrieve the fuzzy system.
    """
    return HIVExpertV2Model().get_model()

if __name__ == '__main__':
    # Local test of the wrapped system
    expert = HIVExpertV2Model()
    FS = expert.get_model()
    
    # Set initial state (in std log space)
    FS.set_variable('T1', 5.0)
    FS.set_variable('T1_inf', 3.0)
    FS.set_variable('T2', -3.0)
    FS.set_variable('T2_inf', -3.0)
    FS.set_variable('V', 5.0)
    FS.set_variable('E', -5.0)
    FS.set_variable('Action', 0)
    
    steps = 50
    dynamics = {var: [FS._variables.get(var, 0.0)] for var in ['T1', 'T2', 'T1_inf', 'T2_inf', 'V', 'E']}
    
    for T in np.linspace(0, 1, steps):
        new_values = FS.Sugeno_inference()
        FS._variables.update(new_values)
        for var in dynamics.keys():
            if var in new_values:
                dynamics[var].append(new_values[var])
            else:
                dynamics[var].append(dynamics[var][-1])

    figure, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    
    ax = axes[0]
    ax.plot(range(steps+1), dynamics['T1'])
    ax.plot(range(steps+1), dynamics['T1_inf'])
    ax.set_ylabel("Level (Std Log)")
    ax.set_xlabel("Time")
    ax.legend(['T1', 'T1_inf'], loc='lower right', framealpha=1.0)

    ax = axes[1]
    ax.plot(range(steps+1), dynamics['T2'])
    ax.plot(range(steps+1), dynamics['T2_inf'])
    ax.set_xlabel('Time')
    ax.set_ylabel('Level (Std Log)')
    ax.legend(['T2', 'T2_inf'], loc='lower right', framealpha=1.0)

    ax = axes[2]
    ax.plot(range(steps+1), dynamics['V'])
    ax.plot(range(steps+1), dynamics['E'])
    ax.set_ylabel("Level (Std Log)")
    ax.set_xlabel("Time")
    ax.legend(['V', 'E'], loc='lower right', framealpha=1.0)

    plt.tight_layout()
    plt.show()
