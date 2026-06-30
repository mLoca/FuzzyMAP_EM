from simpful import *
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
class HIVExpertNewModel:
    def __init__(self):
        #Define fuzzy system
        self.FS = FuzzySystem(show_banner=False)
        #Define rule list
        RULES = []
        #Define crisp output values
        #self.FS.set_crisp_output_value('Off', 0)
        #self.FS.set_crisp_output_value('On', 1)
#
        ## Variable-specific crisp output values
        #self.FS.set_crisp_output_value('T1_Depleted', -4.805361)
        #self.FS.set_crisp_output_value('T1_Partially_depleted', -2.453578)
        #self.FS.set_crisp_output_value('T1_Abundant', 3.033915)
#
        #self.FS.set_crisp_output_value('T1_inf_Depleted', -4.214460)
        #self.FS.set_crisp_output_value('T1_inf_Partially_depleted', -2.407191)
        #self.FS.set_crisp_output_value('T1_inf_Abundant', 1.809770)
#
        #self.FS.set_crisp_output_value('T2_Low', -2.994877)
        #self.FS.set_crisp_output_value('T2_High', 2.770016)
#
        #self.FS.set_crisp_output_value('T2_inf_Low', -3.030424)
        #self.FS.set_crisp_output_value('T2_inf_High', 5.275341)
#
        #self.FS.set_crisp_output_value('V_Low', -3.200451)
        #self.FS.set_crisp_output_value('V_High', 1.953678)
#
        #self.FS.set_crisp_output_value('E_Low', -3.610834)
        #self.FS.set_crisp_output_value('E_High', 21.099048)
#
        #Fuzzy rule definition for T1
        T1_0 = FuzzySet(points=[[-4.805361, 1], [-4.021433, 1], [-3.237506, 0], [3.033915, 0]], term='Depleted')
        T1_1 = FuzzySet(points=[[-4.805361, 0], [-4.021433, 0], [-3.237506, 1], [-1.669651, 1], [0.682132, 0], [3.033915, 0]], term='Partially_depleted')
        T1_2 = FuzzySet(points=[[-4.805361, 0], [-1.669651, 0], [0.682132, 1], [3.033915, 1]], term='Abundant')
        self.FS.add_linguistic_variable('T1', LinguisticVariable([T1_0,T1_1,T1_2], concept='T1'))
        #Fuzzy rule definition for T1_inf
        T1_inf_0 = FuzzySet(points=[[-4.21446, 1], [-3.612037, 1], [-3.009614, 0], [1.80977, 0]], term='Depleted')
        T1_inf_1 = FuzzySet(points=[[-4.21446, 0], [-3.612037, 0], [-3.009614, 1], [-1.804768, 1], [0.002501, 0], [1.80977, 0]], term='Partially_depleted')
        T1_inf_2 = FuzzySet(points=[[-4.21446, 0], [-1.804768, 0], [0.002501, 1], [1.80977, 1]], term='Abundant')
        self.FS.add_linguistic_variable('T1_inf', LinguisticVariable([T1_inf_0,T1_inf_1,T1_inf_2], concept='T1_inf'))
        #Fuzzy rule definition for T2
        T2_0 = FuzzySet(points=[[-2.994877, 1], [1.617038, 1], [2.193527, 0], [2.770016, 0]], term='Low')
        T2_1 = FuzzySet(points=[[-2.994877, 0], [1.617038, 0], [2.193527, 1], [2.770016, 1]], term='High')
        self.FS.add_linguistic_variable('T2', LinguisticVariable([T2_0,T2_1], concept='T2'))
        #Fuzzy rule definition for T2_inf
        T2_inf_0 = FuzzySet(points=[[-3.030424, 1], [3.614188, 1], [4.444764, 0], [5.275341, 0]], term='Low')
        T2_inf_1 = FuzzySet(points=[[-3.030424, 0], [3.614188, 0], [4.444764, 1], [5.275341, 1]], term='High')
        self.FS.add_linguistic_variable('T2_inf', LinguisticVariable([T2_inf_0,T2_inf_1], concept='T2_inf'))
        #Fuzzy rule definition for V
        V_0 = FuzzySet(points=[[-3.200451, 1], [-2.685038, 1], [1.438265, 0], [1.953678, 0]], term='Low')
        V_1 = FuzzySet(points=[[-3.200451, 0], [-2.685038, 0], [1.438265, 1], [1.953678, 1]], term='High')
        self.FS.add_linguistic_variable('V', LinguisticVariable([V_0,V_1], concept='V'))
        #Fuzzy rule definition for E
        E_0 = FuzzySet(points=[[-3.610834, 1], [-3.363736, 1], [20.851949, 0], [21.099048, 0]], term='Low')
        E_1 = FuzzySet(points=[[-3.610834, 0], [-3.363736, 0], [20.851949, 1], [21.099048, 1]], term='High')
        self.FS.add_linguistic_variable('E', LinguisticVariable([E_0,E_1], concept='E'))
        #Fuzzy rule definition for T1p
        T1p_0 = FuzzySet(points=[[-4.805361, 1], [-4.021433, 1], [2.249987, 0], [3.033915, 0]], term='Low')
        T1p_1 = FuzzySet(points=[[-4.805361, 0], [-4.021433, 0], [2.249987, 1], [3.033915, 1]], term='High')
        self.FS.add_linguistic_variable('T1p', LinguisticVariable([T1p_0,T1p_1], concept='T1p'))
        #Fuzzy rule definition for T2p
        T2p_0 = FuzzySet(points=[[-2.994877, 1], [-2.418388, 1], [2.193527, 0], [2.770016, 0]], term='Low')
        T2p_1 = FuzzySet(points=[[-2.994877, 0], [-2.418388, 0], [2.193527, 1], [2.770016, 1]], term='High')
        self.FS.add_linguistic_variable('T2p', LinguisticVariable([T2p_0,T2p_1], concept='T2p'))
        #Fuzzy rule definition for e1
        e1_0 = FuzzySet(points=[[0,1],[0.2,1],[0.8,0],[1,0]], term='Off')
        e1_1 = FuzzySet(points=[[0,0],[0.2,0],[0.8,1],[1,1]], term='On')
        self.FS.add_linguistic_variable('e1', LinguisticVariable([e1_0,e1_1], concept='e1'))
        #Fuzzy rule definition for e2
        e2_0 = FuzzySet(points=[[0,1],[0.2,1],[0.8,0],[1,0]], term='Off')
        e2_1 = FuzzySet(points=[[0,0],[0.2,0],[0.8,1],[1,1]], term='On')
        self.FS.add_linguistic_variable('e2', LinguisticVariable([e2_0,e2_1], concept='e2'))
        #Fuzzy rule definition for next_E
        next_E_0 = FuzzySet(points=[[-3.610834, 1], [-3.363736, 1], [20.851949, 0], [21.099048, 0]], term='Low')
        next_E_1 = FuzzySet(points=[[-3.610834, 0], [-3.363736, 0], [20.851949, 1], [21.099048, 1]], term='High')
        self.FS.add_linguistic_variable('next_E', LinguisticVariable([next_E_0,next_E_1], concept='E'))

        #Fuzzy rule definition for next_V
        next_V_0 = FuzzySet(points=[[-3.200451, 1], [-2.685038, 1], [1.438265, 0], [1.953678, 0]], term='Low')
        next_V_1 = FuzzySet(points=[[-3.200451, 0], [-2.685038, 0], [1.438265, 1], [1.953678, 1]], term='High')
        self.FS.add_linguistic_variable('next_V', LinguisticVariable([next_V_0,next_V_1], concept='V'))

        #Define rules for T1
        #RULES.append('IF (T1p IS High) THEN (T1 IS T1_Abundant)')
        #RULES.append('IF (T1p IS Low) THEN (T1 IS T1_Depleted)')
        #RULES.append('IF (V IS High) THEN (T1 IS T1_Partially_depleted)')

        
        #Define rules for T1_inf
        RULES.append('IF (V IS High) AND (T1 IS Abundant) THEN (T1_inf IS Abundant)')
        RULES.append('IF (V IS High) AND (T1 IS Partially_depleted) THEN (T1_inf IS Partially_depleted)')
        RULES.append('IF (V IS High) AND (T1 IS Depleted) THEN (T1_inf IS Depleted)')
        RULES.append('IF (V IS Low) THEN (T1_inf IS Depleted)')
        RULES.append('IF (E IS High) THEN (T1_inf IS Depleted)')
        RULES.append('IF (e1 IS On) THEN (T1_inf IS Depleted)')

        #Define rules for T2
        #RULES.append('IF (T2p IS High) THEN (T2 IS T2_High)')
        #RULES.append('IF (T2p IS Low) THEN (T2 IS T2_Low)')
        #RULES.append('IF (V IS High) THEN (T2 IS T2_Low)')
        
        #Define rules for T2_inf
        RULES.append('IF (V IS High) THEN (T2_inf IS High)')
        RULES.append('IF (T2 IS Low) THEN (T2_inf IS Low)')
        RULES.append('IF (V IS Low) THEN (T2_inf IS Low)')
        RULES.append('IF (E IS High) THEN (T2_inf IS Low)')
        RULES.append('IF (e1 IS On) THEN (T2_inf IS Low)')
        
        #Define rules for V
        RULES.append('IF (T1_inf IS Abundant) OR (T1_inf IS Partially_depleted) OR (T2_inf IS High) THEN (V IS High)')
        RULES.append('IF (T1_inf IS Depleted) AND (T2_inf IS Low) THEN (V IS Low)')
        RULES.append('IF (e2 IS On) THEN (V IS Low)')
        
        #Define rules for E
        RULES.append('IF (T1_inf IS Abundant) THEN (E IS Low)')
        RULES.append('IF (T1_inf IS Partially_depleted) THEN (E IS High)')
        RULES.append('IF (T1_inf IS Depleted) THEN (E IS Low)')
        RULES.append('IF (V IS High) THEN (E IS Low)')
        RULES.append('IF (V IS Low) THEN (E IS High)')
        
        #add fuzzy rules
        self.FS.add_rules(RULES)

    def get_model(self):
        return self.FS

if __name__ == "__main__":
    expert = HIVExpertNewModel()
    FS = expert.get_model()
    #Define initial state for T1
    FS.set_variable('T1', 3.033915)
    #Define initial state for T1_inf
    FS.set_variable('T1_inf', -0.599922)
    #Define initial state for T2
    FS.set_variable('T2', 0.464059)
    #Define initial state for T2_inf
    FS.set_variable('T2_inf', 1.953035)
    #Define initial state for V
    FS.set_variable('V', 0.922852)
    #Define initial state for E
    FS.set_variable('E', 6.273119)
    #Define initial state for T1p
    FS.set_variable('T1p', 3.033915)
    #Define initial state for T2p
    FS.set_variable('T2p', 1.040548)
    #Define initial state for e1
    FS.set_variable('e1', 0.7)
    #Define initial state for e2
    FS.set_variable('e2', 0.3)
    # Set number of inference steps and save initial state
    steps=50
    dynamics = {var: [] for var in ['T1', 'T1_inf', 'T2', 'T2_inf', 'V', 'E']}
    for var in dynamics.keys():
        dynamics[var].append(FS._variables[var])
    # Perform Sugeno inference and save results
    for T in np.linspace(0, 1, steps):
        new_values = FS.Sugeno_inference()
        
        # Map next_ states back to current states for the next iteration
        update_dict = {}
        for k, v in new_values.items():
            if k.startswith("next_"):
                update_dict[k.replace("next_", "")] = v
            else:
                update_dict[k] = v
                
        FS._variables.update(update_dict)
        # Perturbations can be added here using the set_variable method
        for var in dynamics.keys():
            dynamics[var].append(FS._variables[var])
    #Plotting dynamics
    T1 = dynamics['T1']
    T2 = dynamics['T2']
    T1_inf = dynamics['T1_inf']
    T2_inf = dynamics['T2_inf']
    V = dynamics['V']
    E = dynamics['E']
    figure, axes = plt.subplots(nrows=1, ncols=3)
    ax = axes[0]
    ax.plot(range(steps+1), T1)
    ax.plot(range(steps+1), T1_inf)
    ax.set_ylabel("Level")
    ax.set_xlabel("Time")
    #ax.set_ylim(0,1100)
    ax.legend(['T1', 'T1_inf'], loc='lower right',framealpha=1.0)

    ax = axes[1]
    ax.plot(range(steps+1), T2)
    ax.plot(range(steps+1), T2_inf)
    ax.set_xlabel('Time')
    ax.set_ylabel('Level')
    #ax.set_ylim(0,1100)
    ax.legend(['T2', 'T2_inf'], loc='lower right',framealpha=1.0)


    ax = axes[2]
    ax.plot(range(steps+1), V)
    ax.plot(range(steps+1), E)
    ax.set_ylabel("Level")
    ax.set_xlabel("Time")
    #ax.set_ylim(0,1100)
    ax.legend(['V', 'E'], loc='lower right',framealpha=1.0)


    plt.show()
