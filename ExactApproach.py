
# get_ipython().system('pip install pulp')

from pulp import LpProblem, LpMinimize, LpVariable, lpSum, PULP_CBC_CMD, value, LpStatus
import numpy as np
import pandas as pd
import time

# ******************** Load dataset *********************** #
train = pd.read_csv('./iris.csv', skipinitialspace=True)
numeric_col = train.select_dtypes(include=['float64', 'int64'])
X = numeric_col.to_numpy()


# # Scale data
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# X.shape
# X=X[:,:1]


X


# ******************** MILP CBC Function *********************** #
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, PULP_CBC_CMD, value
import numpy as np

def exact_milp_clustering(X, k):
    # Dimensions
    n_points, n_dim = X.shape
    
    # Create the MILP problem
    prob = LpProblem("L1 Clustering", LpMinimize)
    
    # Decision variables
    # Assignment variables: y[(i, j)] = 1 if point i is assigned to cluster j
    y = LpVariable.dicts("y", ((i, j) for i in range(n_points) for j in range(k)), cat='Binary')
    
    # Distance variables: z[i] represents the L1 distance for point i to its assigned cluster center
    z = {i: LpVariable(f"z_{i}", lowBound=0, cat="Continuous") for i in range(n_points)}
    
    # Absolute difference variables: d_ijl[(i, j, l)] for absolute differences between point i and cluster center j in dimension l
    d_ijl = LpVariable.dicts("d_ijl", ((i, j, l) for i in range(n_points) for j in range(k) for l in range(n_dim)), lowBound=0, cat='Continuous')
    
    # Cluster center variables: c[(j, l)] represents the l-th coordinate of the j-th cluster center
    c = LpVariable.dicts("c", ((j, l) for j in range(k) for l in range(n_dim)), lowBound=0, cat='Continuous')

    # Big-M constant, calculated as the sum of dimension-specific ranges\
    M_l=[(np.max(X[:, l]) - X[:, l].min()) for l in range(n_dim)]
    #M_l = [np.max(np.abs(X[:, l].max() - X[:, l].min())) for l in range(n_dim)]
    M = sum(M_l)  # Use the total range as the Big-M constant
    
    # Objective: Minimize the sum of L1 distances z[i]
    prob += lpSum(z[i] for i in range(n_points)), "Total L1 Distance"
    
    # Constraints
    # 1. Each point must be assigned to exactly one cluster
    for i in range(n_points):
        prob += lpSum(y[(i, j)] for j in range(k)) == 1, f"Assign_to_one_center_{i}"
    
    # 2. Calculate the L1 distance only for assigned clusters using Big-M
    for i in range(n_points):
        for j in range(k):
            prob += z[i] >= lpSum(d_ijl[(i, j, l)] for l in range(n_dim)) - M * (1 - y[(i, j)]), f"L1_distance_{i}_{j}"
    
    # 3. Absolute difference constraints: positive direction
    for i in range(n_points):
        for j in range(k):
            for l in range(n_dim):
                prob += d_ijl[(i, j, l)] >= X[i, l] - c[(j, l)], f"Abs_Pos_{i}_{j}_{l}"
    
    # 4. Absolute difference constraints: negative direction
    for i in range(n_points):
        for j in range(k):
            for l in range(n_dim):
                prob += d_ijl[(i, j, l)] >= c[(j, l)] - X[i, l], f"Abs_Neg_{i}_{j}_{l}"
    
    # 5. Bounding Box for Centroids: Ensure each c[j][l] lies within the min and max range of X[:, l]
    for j in range(k):
        for l in range(n_dim):
            prob += c[(j, l)] >= np.min(X[:, l]), f"Centroid_Lower_Bound_{j}_{l}"
            prob += c[(j, l)] <= np.max(X[:, l]), f"Centroid_Upper_Bound_{j}_{l}"
    
    # Solve the problem using the CBC solver
    prob.solve(PULP_CBC_CMD())
    
    # Check solver status
    if prob.status != 1:  # 1 means "Optimal"
        print("Solver Status:", LpStatus[prob.status])
        return [], 0
    
    # Extract the results
    selected_centers = [j for j in range(k) if any(value(y[(i, j)]) > 0.5 for i in range(n_points))]
    total_l1_distance = value(prob.objective)
    
    return selected_centers, total_l1_distance


# print("Running...") // Debugging


# ******************** Function to run experiments *********************** #
def run_experiments(X, k_list, n_list):
    results = []
    
    for n in n_list:
        X_sample = X[:n]  # Select first n points as sample
        for k in k_list:
            # Run exact MILP clustering
            start_time = time.time()
            selected_centers, total_l1_distance = exact_milp_clustering(X_sample, k)
            elapsed_time = time.time() - start_time
            
            # Record the results
            results.append({
                'n': n,
                'k': k,
                'method': 'CBC MILP',
#               'selected_centers': selected_centers,
                'total_l1_distance': total_l1_distance,
                'time_taken': elapsed_time,
            })
    
    # Convert results to DataFrame 
    return pd.DataFrame(results)


# print("Running...") // Debugging


# Experiment values
k_list = [2,3,4]  # Example values for k
n_list = [15,20]

# Run the experiments
results_df = run_experiments(X, k_list, n_list)

# Output the experiment results
print(results_df)




# ******************** Ploting *********************** #
import matplotlib.pyplot as plt
import seaborn as sns

# Plot the results: Time Taken vs Number of Data Points (n)
def plot_results(results_df):
    plt.figure(figsize=(14, 6))

    # Plot 1: Time Taken vs Number of Data Points (n) for different k
    plt.subplot(1, 2, 1)
    sns.lineplot(data=results_df, x='n', y='time_taken', hue='k', marker='o')
    plt.title('Time Taken by CBC MILP Solver vs Number of Data Points')
    plt.xlabel('Number of Data Points (n)')
    plt.ylabel('Time Taken (seconds)')
    plt.legend(title='k (clusters)', loc='upper left')
    
    # Plot 2: Total L1 Distance vs Number of Data Points (n) for different k
    plt.subplot(1, 2, 2)
    sns.lineplot(data=results_df, x='n', y='total_l1_distance', hue='k', marker='o')
    plt.title('Total L1 Distance vs Number of Data Points')
    plt.xlabel('Number of Data Points (n)')
    plt.ylabel('Total L1 Distance')
    plt.legend(title='k (clusters)', loc='upper left')
    
    plt.tight_layout()
    plt.show()

# Call the plotting function after running the experiments
plot_results(results_df)







