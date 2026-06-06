import numpy as np
from simpful import *

from hiv_simulator import HIVSimulator

class HIVExpert5DModel:
    def __init__(self):
        """
        5-Dimensional Takagi-Sugeno (TS) Fuzzy Inference System.
        Inputs: Action, T1 (CD4+), T2 (Macrophages), V (Viral Load), E (Immune Response)
        Outputs: next_T1, next_T2, next_V, next_E
        """
        self.FS = FuzzySystem(show_banner=False)

        # ==========================================
        # 1. ANTECEDENTS (Current State & Actions)
        # ==========================================
        
        # Treatment Action (0=None, 1=RTI, 2=PI, 3=HAART)
        A_none = FuzzySet(function=Triangular_MF(a=0.0, b=0.0, c=0.25), term="none")
        A_weak = FuzzySet(function=Trapezoidal_MF(a=0.25, b=0.5, c=2.5, d=2.55), term="weak")
        A_strong = FuzzySet(function=Triangular_MF(a=0.75, b=3.0, c=3.10), term="strong")
        self.FS.add_linguistic_variable("Action", LinguisticVariable([A_none, A_weak, A_strong], universe_of_discourse=[0, 3]))

        # Current Viral Load (V) - Assuming Log10 Space
        V_low = FuzzySet(function=Trapezoidal_MF(a=-4.2, b=-4.2, c=-0.2, d=0.2), term="low")
        V_high = FuzzySet(function=Trapezoidal_MF(a=-0.2, b=0.2, c=2.1, d=2.1), term="high")
        self.FS.add_linguistic_variable("V", LinguisticVariable([V_low, V_high], universe_of_discourse=[-4.2, 2.1]))

        # Current T1 Cells (CD4+) - Assuming Log10 Space
        T1_dep = FuzzySet(function=Trapezoidal_MF(a=-4.5, b=-4.5, c=-0.2, d=0.2), term="depleted")
        T1_hlt = FuzzySet(function=Trapezoidal_MF(a=-0.2, b=0.2, c=2.2, d=2.2), term="healthy")
        self.FS.add_linguistic_variable("T1", LinguisticVariable([T1_dep, T1_hlt], universe_of_discourse=[-4.5, 2.2]))

        # Current T2 Cells (Macrophages) - Assuming Log10 Space
        T2_dep = FuzzySet(function=Trapezoidal_MF(a=-2.6, b=-2.6, c=-0.1, d=0.1), term="depleted")
        T2_hlt = FuzzySet(function=Trapezoidal_MF(a=-0.1, b=0.1, c=1.8, d=1.8), term="healthy")
        self.FS.add_linguistic_variable("T2", LinguisticVariable([T2_dep, T2_hlt], universe_of_discourse=[-2.6, 1.8]))

        # Current Immune Response (E) - Assuming Log10 Space
        E_weak = FuzzySet(function=Trapezoidal_MF(a=-3.5, b=-3.5, c=-0.5, d=0.1), term="weak")
        E_strong = FuzzySet(function=Trapezoidal_MF(a=-0.1, b=0.5, c=4.2, d=4.2), term="strong")
        self.FS.add_linguistic_variable("E", LinguisticVariable([E_weak, E_strong], universe_of_discourse=[-3.5, 4.2]))

        # ==========================================
        # 2. SUGENO CONSEQUENT FUNCTIONS (Mathematical Transitions)
        # ==========================================
        
        # Next Viral Load (V)
        self.FS.set_output_function("fun1", "V + (-0.94 - V) * 0.90" )               # HAART
        self.FS.set_output_function("fun2", "V + (-0.56 - V) * 0.91")            # Mono-therapy
        self.FS.set_output_function("fun3", "V + (1.04 - V) * 0.81")    # Unchecked growth

        # Next CD4+ T Cells (T1)
        self.FS.set_output_function("fun4", "T1 + (2.09 - T1) * 0.14")
        self.FS.set_output_function("fun5", "T1 + (2.03 - T1) * 0.14")
        self.FS.set_output_function("fun6", "T1 + (0.14 - T1) * 0.86")

        # Next T2 Cells
        self.FS.set_output_function("fun7", "T2 + (0.90 - T2) * 0.72")
        self.FS.set_output_function("fun8", "T2 + (0.68 - T2) * 0.80")
        self.FS.set_output_function("fun9", "T2 + (-0.81 - T2) * 1.06")

        # Next Cytotoxic T Cells (E)
        self.FS.set_output_function("fun10", "E + (6.52 - E) * 0.07")
        self.FS.set_output_function("fun11", "E + (6.35 - E) * 0.07")
        self.FS.set_output_function("fun12", "E + (-0.70 - E) * 0.07")             # High virus exhausts immune system

        # ==========================================
        # 3. FUZZY CLINICAL RULES 
        # ==========================================
        rules = [
            # 1. HAART (Action 3) with a functioning immune system leads to full recovery
            "IF (Action IS strong) AND (E IS strong) THEN (next_V IS fun1)",
            "IF (Action IS strong) AND (E IS strong) THEN (next_T1 IS fun4)",
            "IF (Action IS strong) AND (E IS strong) THEN (next_T2 IS fun7)",
            "IF (Action IS strong) AND (E IS strong) THEN (next_E IS fun10)",
            # 2. HAART (Action 3) on a severely depleted system has slower recovery
            "IF (Action IS strong) AND (E IS weak) THEN (next_V IS fun2)",
            "IF (Action IS strong) AND (E IS weak) THEN (next_T1 IS fun5)",
            "IF (Action IS strong) AND (E IS weak) THEN (next_T2 IS fun8)",
            "IF (Action IS strong) AND (E IS weak) THEN (next_E IS fun10)",
            # 3. Mono-therapy (Action 1 or 2) holds the line but doesn't crush the virus
            "IF (Action IS weak) THEN (next_V IS fun2)",
            "IF (Action IS weak) THEN (next_T1 IS fun5)",
            "IF (Action IS weak) THEN (next_T2 IS fun8)",
            "IF (Action IS weak) THEN (next_E IS fun11)",
            
            # 4. No Therapy (Action 0) while Virus is High leads to widespread depletion
            "IF (Action IS none) AND (V IS high) THEN (next_V IS fun3)",
            "IF (Action IS none) AND (V IS high) THEN (next_T1 IS fun6)",
            "IF (Action IS none) AND (V IS high) THEN (next_T2 IS fun9)",
            "IF (Action IS none) AND (V IS high) THEN (next_E IS fun12)",
            
            # 5. No Therapy (Action 0) while Virus is Low allows the system to remain temporarily stable
            "IF (Action IS none) AND (V IS low) THEN (next_V IS fun2)",
            "IF (Action IS none) AND (V IS low) THEN (next_T1 IS fun5)",
            "IF (Action IS none) AND (V IS low) THEN (next_T2 IS fun8)",
            "IF (Action IS none) AND (V IS low) THEN (next_E IS fun11)"
        ]
        self.FS.add_rules(rules)

    def predict_next_state(self, action, t1, t2, v, e):
        """
        Evaluates the 5D Takagi-Sugeno fuzzy model safely clamped to empirical bounds.
        """
        self.FS.set_variable("Action", action)
        self.FS.set_variable("T1", max(-4.49, min(2.19, t1)))
        self.FS.set_variable("T2", max(-2.59, min(1.79, t2)))
        self.FS.set_variable("V",  max(-4.19, min(2.09, v)))
        self.FS.set_variable("E",  max(-3.49, min(4.19, e)))

        res = self.FS.Sugeno_inference(["next_T1", "next_T2", "next_V", "next_E"])
        
        return res["next_T1"], res["next_T2"], res["next_V"], res["next_E"]

    def get_model(self):
        return self.FS

    def evaluate_fuzzy_accuracy(self, n_test=1000, seed=42):
        """
        Evaluates the predictive accuracy of the Fuzzy Expert Model against 
        the ground-truth HIV ODE Simulator using L1 Error (Mean Absolute Error).
        """
        np.random.seed(seed)
        
        # Initialize both systems
        # podmp=True applies the T1, T2, V, E mask automatically
        env = HIVSimulator(podmp=True, logspace=True)
        expert = HIVExpert5DModel()
        
        # Dictionaries to track the L1 error for each biological marker
        l1_errors = {"T1": [], "T2": [], "V": [], "E": []}
        
        print(f"Generating {n_test} clinical transitions for evaluation...")
        
        for _ in range(n_test):
            # 1. Get a biologically valid initial state
            state = env.reset(perturb_params=True)
            current_T1, current_T2, current_V, current_E = state
            
            # 2. Pick a random treatment action (0, 1, 2, or 3)
            action = np.random.randint(0, env.num_actions)
            
            # 3. Ground Truth: Step the actual ODE simulator
            true_next_state, _, _, _ = env.step(action)
            true_T1, true_T2, true_V, true_E = true_next_state
            
            # 4. Prediction: Ask the Fuzzy Expert to predict the next state
            pred_T1, pred_T2, pred_V, pred_E = expert.predict_next_state(
                action, current_T1, current_T2, current_V, current_E
            )
            
            # 5. Calculate Absolute Errors
            l1_errors["T1"].append(abs(true_T1 - pred_T1))
            l1_errors["T2"].append(abs(true_T2 - pred_T2))
            l1_errors["V"].append(abs(true_V - pred_V))
            l1_errors["E"].append(abs(true_E - pred_E))
            
        # Calculate the mean across all tested samples
        mae_T1 = np.mean(l1_errors["T1"])
        mae_T2 = np.mean(l1_errors["T2"])
        mae_V  = np.mean(l1_errors["V"])
        mae_E  = np.mean(l1_errors["E"])
        
        overall_mae = np.mean([mae_T1, mae_T2, mae_V, mae_E])
        
        print("\n=== Fuzzy Expert Prior Accuracy ===")
        print("Metric: Mean Absolute Error (L1) in log10 space.")
        print("Lower is better (0.0 indicates perfect alignment with the ODE).")
        print("-" * 55)
        print(f"CD4+ Cells (T1) Error:      {mae_T1:.4f}")
        print(f"Macrophages (T2) Error:     {mae_T2:.4f}")
        print(f"Viral Load (V) Error:       {mae_V:.4f}")
        print(f"Immune Response (E) Error:  {mae_E:.4f}")
        print("-" * 55)
        print(f"OVERALL SYSTEM MAE:         {overall_mae:.4f}")
        
        return overall_mae

    #TODO: change position:
    def find_empirical_bounds(self, n_samples=1000):
        env = HIVSimulator(podmp=True, logspace=True)
        
        # Store all observations
        data = {"T1": [], "T2": [], "V": [], "E": []}
        
        for _ in range(n_samples):
            state = env.reset(perturb_params=True)
            # Randomly step through the environment
            for _ in range(50):
                action = np.random.randint(0, env.num_actions)
                next_state, _, done, _ = env.step(action)
                
                data["T1"].append(next_state[0])
                data["T2"].append(next_state[1])
                data["V"].append(next_state[2])
                data["E"].append(next_state[3])
                
                if done: break

        print(f"{'Marker':<5} | {'Min':<8} | {'Max':<8} | {'Mean':<8}")
        print("-" * 35)
        for key, vals in data.items():
            print(f"{key:<5} | {np.min(vals):<8.2f} | {np.max(vals):<8.2f} | {np.mean(vals):<8.2f}")

# ==========================================
# TEST AND SIMULATE
# ==========================================
if __name__ == "__main__":
    expert = HIVExpert5DModel()
    
    # Format: Action, T1, T2, V, E
    scenarios = [
        {"name": "HAART on Healthy Patient", "act": 3, "t1": 5.0, "t2": 3.0, "v": 3.0, "e": 4.0},
        {"name": "HAART on Critical Patient", "act": 3, "t1": 1.0, "t2": 0.5, "v": 6.5, "e": 1.0},
        {"name": "Mono-therapy",              "act": 1, "t1": 3.0, "t2": 2.0, "v": 5.0, "e": 2.5},
        {"name": "No Therapy (High Virus)",   "act": 0, "t1": 4.0, "t2": 2.5, "v": 7.0, "e": 3.0}
    ]

    print(f"{'Scenario':<26} | {'Act':<3} | {'V(In)':<5} -> {'V(Out)':<6} | {'T1(In)':<6} -> {'T1(Out)':<7} | {'T2(In)':<6} -> {'T2(Out)':<7} | {'E(In)':<5} -> {'E(Out)':<6}")
    print("-" * 115)
    
    for s in scenarios:
        nT1, nT2, nV, nE = expert.predict_next_state(s["act"], s["t1"], s["t2"], s["v"], s["e"])
        print(f"{s['name']:<26} | {s['act']:<3} | {s['v']:<5.2f} -> {nV:<6.2f} | {s['t1']:<6.2f} -> {nT1:<7.2f} | {s['t2']:<6.2f} -> {nT2:<7.2f} | {s['e']:<5.2f} -> {nE:<6.2f}")

    #expert.find_empirical_bounds(n_samples=1000)
    expert.evaluate_fuzzy_accuracy()