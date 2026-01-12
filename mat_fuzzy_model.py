from simpful import *


def create_fuzzy_model():
    # variable with 3 gaussian membership functions
    FS = FuzzySystem()

    FS.add_linguistic_variable(
        name="test_result",
        LV=LinguisticVariable(
            FS_list=[
                FuzzySet(function=Gaussian_MF(mu=0, sigma=0.18), term="negative"),
                FuzzySet(function=Gaussian_MF(mu=0.5, sigma=0.18), term="medium"),
                FuzzySet(function=Gaussian_MF(mu=1, sigma=0.18), term="positive"),
            ],
            universe_of_discourse=[0, 1],
            concept="test_result"
        ))

    FS.add_linguistic_variable(
        name="symptoms",
        LV=LinguisticVariable(
            FS_list=[
                FuzzySet(function=Gaussian_MF(mu=0, sigma=0.18), term="negative"),
                FuzzySet(function=Gaussian_MF(mu=0.5, sigma=0.18), term="medium"),
                FuzzySet(function=Gaussian_MF(mu=1, sigma=0.18), term="positive"),
            ],
            universe_of_discourse=[0, 1],
            concept="symptoms"
        )
    )

    FS.add_linguistic_variable(
        name="action",
        LV=LinguisticVariable(
            FS_list=[
                FuzzySet(function=Triangular_MF(a=0, b=0, c=1), term="wait"),
                FuzzySet(function=Triangular_MF(a=0, b=1, c=1), term="treat"),
            ],
            universe_of_discourse=[0, 1],
            concept="action"
        )
    )

    FS.set_output_function("fun1", "0.008*test_result+ 0.157*symptoms+ 0.139*action +0.055")
    FS.set_output_function("fun2", "0.455*test_result+ 0.000*symptoms+ 0.000*action +0.220")
    FS.set_output_function("fun3", "0.529*test_result+ 0.632*symptoms+ 0.369*action +0.001")
    FS.set_output_function("fun4", "0.480*test_result+ 0.157*symptoms+ 0.000*action +0.500")
    FS.set_output_function("fun5", "0.000*test_result+ 0.297*symptoms+ 0.071*action +0.293")
    FS.set_output_function("fun6", "0.234*test_result+ 0.202*symptoms+ 0.074*action +0.034")

    FS.add_rules(rules=["IF (test_result IS negative) AND (action IS wait) THEN (next_test IS fun1)",
                        "IF (test_result IS medium) AND (symptoms IS negative) THEN (next_test IS fun2)",
                        "IF (test_result IS negative) AND (action IS wait) THEN (next_test IS fun2)",
                        "IF (test_result IS negative) AND (symptoms IS negative) THEN (next_test IS fun2)",
                        "IF (test_result IS negative) AND (symptoms IS negative) AND (action IS wait) THEN (next_test IS fun2)",
                        "IF (test_result IS medium) AND (symptoms IS medium) AND (action IS wait) THEN (next_test IS fun3)",
                        "IF (action IS treat) THEN (next_test IS fun1)",
                        "IF (test_result IS medium) AND (symptoms IS negative) AND (action IS treat) THEN (next_test IS fun1)",
                        "IF (test_result IS medium) AND (symptoms IS negative) AND (action IS wait) THEN (next_test IS fun1)",
                        "IF (test_result IS positive) AND (symptoms IS negative) AND (action IS wait) THEN (next_test IS fun3)",
                        "IF (test_result IS negative) AND (action IS treat) THEN (next_test IS fun1)",
                        "IF (test_result IS negative) AND (symptoms IS medium) AND (action IS wait) THEN (next_test IS fun1)",
                        "IF (test_result IS positive) AND (symptoms IS positive) AND (action IS treat) THEN (next_test IS fun1)",
                        "IF (test_result IS negative) AND (symptoms IS positive) AND (action IS wait) THEN (next_test IS fun3)",
                        "IF (test_result IS medium) AND (symptoms IS positive) AND (action IS wait) THEN (next_test IS fun1)",
                        "IF (test_result IS medium) AND (symptoms IS medium) AND (action IS treat) THEN (next_test IS fun2)",
                        "IF (test_result IS negative) AND (symptoms IS positive) AND (action IS treat) THEN (next_test IS fun1)",
                        "IF (symptoms IS medium) AND (action IS wait) THEN (next_test IS fun2)",
                        "IF (action IS wait) THEN (next_test IS fun3)",
                        "IF (test_result IS medium) THEN (next_test IS fun2)",
                        "IF (test_result IS positive) AND (symptoms IS medium) THEN (next_test IS fun2)",
                        "IF (test_result IS negative) THEN (next_test IS fun1)",
                        "IF (symptoms IS negative) THEN (next_test IS fun2)",
                        "IF (symptoms IS positive) THEN (next_test IS fun2)",
                        #AAA
                        "IF (test_result IS negative) AND (action IS wait) THEN (next_symptoms IS fun6)",
                        "IF (test_result IS negative) AND (symptoms IS negative) THEN (next_symptoms IS fun4)",
                        "IF (test_result IS negative) AND (symptoms IS negative) AND (action IS wait) THEN (next_symptoms IS fun5)",
                        "IF (test_result IS medium) AND (symptoms IS medium) AND (action IS wait) THEN (next_symptoms IS fun4)",
                        "IF (action IS treat) THEN (next_symptoms IS fun6)",
                        "IF (test_result IS medium) AND (symptoms IS negative) AND (action IS treat) THEN (next_symptoms IS fun4)",
                        "IF (test_result IS medium) AND (symptoms IS negative) AND (action IS wait) THEN (next_symptoms IS fun5)",
                        "IF (test_result IS negative) AND (action IS treat) THEN (next_symptoms IS fun4)",
                        "IF (test_result IS negative) AND (symptoms IS medium) AND (action IS wait) THEN (next_symptoms IS fun5)",
                        "IF (test_result IS positive) AND (symptoms IS positive) AND (action IS treat) THEN (next_symptoms IS fun4)",
                        "IF (test_result IS negative) AND (symptoms IS positive) AND (action IS wait) THEN (next_symptoms IS fun4)",
                        "IF (test_result IS medium) AND (symptoms IS positive) AND (action IS wait) THEN (next_symptoms IS fun4)",
                        "IF (test_result IS negative) AND (symptoms IS medium) AND (action IS treat) THEN (next_symptoms IS fun5)",
                        "IF (test_result IS medium) AND (symptoms IS medium) AND (action IS treat) THEN (next_symptoms IS fun5)",
                        "IF (test_result IS negative) AND (symptoms IS positive) AND (action IS treat) THEN (next_symptoms IS fun4)",
                        "IF (symptoms IS medium) AND (action IS wait) THEN (next_symptoms IS fun6)",
                        "IF (action IS wait) THEN (next_symptoms IS fun4)",
                        "IF (test_result IS medium) THEN (next_symptoms IS fun6)",
                        "IF (test_result IS positive) AND (symptoms IS medium) THEN (next_symptoms IS fun4)",
                        "IF (test_result IS medium) AND (action IS treat) THEN (next_symptoms IS fun5)",
                        "IF (test_result IS medium) AND (action IS wait) THEN (next_symptoms IS fun4)",
                        "IF (symptoms IS positive) THEN (next_symptoms IS fun5)"
                        ])

    return FS


if __name__ == "__main__":
    #create_fuzzy_model()
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import multivariate_normal

    # Define the parameters
    parameters = [
        {
            'name': 'Healthy',
            'mean': [0.30, 0.30],
            'cov': [[0.007, 0.0], [0.0001, 0.007]]
        },
        {
            'name': 'Sick',
            'mean': [0.42, 0.45],
            'cov': [[0.02, 0.0125], [0.0125, 0.02]]
        },
        {
            'name': 'Critical',
            'mean': [0.625, 0.75],
            'cov': [[0.011, 0.001], [0.001, 0.011]]
        }
    ]

    state_names = ["Healthy", "Sick", "Critical"]
    n_states = len(state_names)
    title_prefix = "Distribution for"

    plt.figure(figsize=(15, 5))

    for s in range(n_states):
        plt.subplot(1, 3, s + 1)

        # Create a 2D grid of points
        x = np.linspace(0, 1, 100)
        y = np.linspace(0, 1, 100)
        X, Y = np.meshgrid(x, y)

        # Create the position array for vectorized pdf calculation
        # This replaces the nested loop for performance, producing the same Z
        pos = np.dstack((X, Y))

        # Get parameters for current state
        mean = parameters[s]['mean']
        cov = parameters[s]['cov']
        rv = multivariate_normal(mean, cov)

        # Compute the PDF on the grid
        Z = rv.pdf(pos)

        plt.contourf(X, Y, Z, levels=20, cmap='viridis')
        plt.title(f"{title_prefix} State {state_names[s]}")
        plt.xlabel("Test Result")
        plt.ylabel("Symptoms")
        plt.colorbar()

    plt.tight_layout()
    plt.show()
    #plt.savefig('mvn_subplots.png')
