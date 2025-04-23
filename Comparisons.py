
import numpy as np
import pandas as pd
import time
from pulp import LpMinimize, LpProblem, LpVariable, lpSum, LpBinary, value,LpStatus
import seaborn as sns
from pulp import PULP_CBC_CMD
# from sklearn.preprocessing import StandardScaler



# ******************** Load dataset *********************** #
train = pd.read_csv('./iris.csv', skipinitialspace=True)
numeric_col = train.select_dtypes(include=['float64', 'int64'])
X = numeric_col.to_numpy()


# In[53]:


# Standardize the data (zero mean, unit variance)
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

X.shape

# print(X_scaled[:500]) 
# X=X[:,:1]

X


# ****************** K-medians heuristic function *************************#
def k_medians(X, k, max_iter=100, tol=1e-4, random_seed=None):
    if k > len(X):
        raise ValueError("k cannot be greater than the number of data points.")
    
    if random_seed:
        np.random.seed(random_seed)
    
    initial_indices = np.random.choice(X.shape[0], k, replace=False)
    centroids = X[initial_indices]
    
    for iteration in range(max_iter):
        distances = np.abs(X[:, np.newaxis] - centroids).sum(axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([np.median(X[labels == i], axis=0) for i in range(k)])
        
        if np.abs(new_centroids - centroids).sum() < tol:
            break
        
        centroids = new_centroids
    
    total_l1_distance = np.sum(np.abs(X - centroids[labels]))
    print(f"Heuristic Total L1 Distance: {total_l1_distance}")  # Debugging output
    return centroids, labels, total_l1_distance


# ******************** MILP CBC Function *********************** #
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, PULP_CBC_CMD, value
import numpy as np

def exact_milp_clustering(X, k, time_limit=None):

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


# ******************** Compare Function *********************** #
def compare_solutions(heuristic_labels, milp_centers, X, k):
    heuristic_clusters = {i: X[heuristic_labels == i] for i in range(k)}
    milp_centroids = X[milp_centers]

    print("\nComparison of Heuristic and MILP Solutions:")
    print("--------------------------------------------------")

    # Ensure number of MILP centroids matches k
    if len(milp_centroids) != k:
        print(f"Warning: Number of MILP centroids ({len(milp_centroids)}) does not match k ({k})!")
        return
    
    for i in range(k):
        print(f"Cluster {i + 1} (Heuristic):")
        print(heuristic_clusters[i])
        print(f"Centroid {i + 1} (MILP):")
        print(milp_centroids[i])
        print("--------------------------------------------------")



# ******************** GAP Calculation *********************** #
def calculate_gap(heuristic_distance, milp_distance):
    if milp_distance == 0:
        return 0.0  # If MILP distance is zero (or both are zero), no gap
    
    if heuristic_distance > milp_distance:
        gap = ((heuristic_distance - milp_distance) / milp_distance) * 100
        return max(gap, 0.0)  # The gap cannot be negative
    else:
        return 0.0  # Return 0% gap if MILP performs worse or equal



# ******************** Experiment Function *********************** #
def run_experiments_with_milp(X, k_list, n_list, time_limit=600, random_seeds=[46], milp_enabled=True):
    results = []
    
    for n in n_list:
        X_sample = X[:n]
        
        for k in k_list:
            if k > n:
                continue
            
            for seed in random_seeds:
                start_time = time.time()
                centroids, labels, total_l1_distance_heuristic = k_medians(X_sample, k, random_seed=seed)
                heuristic_time = time.time() - start_time

                result_entry = {
                    'n': n,
                    'k': k,
                    'method': 'K-medians',
                    'total_l1_distance': total_l1_distance_heuristic,
                    'time_taken': heuristic_time,
                    'seed': seed,
                    'gap (%)': None
                }

                if milp_enabled and n <= 1000:  # Run MILP for small datasets
                    start_time_milp = time.time()
                    milp_centers, total_l1_distance_milp = exact_milp_clustering(X_sample, k, time_limit)
                    milp_time = time.time() - start_time_milp

                    if total_l1_distance_milp is not None:
                        print(f"MILP centers for k={k}: {milp_centers}")  # Log the MILP centers
                        # Compare solutions
                        compare_solutions(labels, milp_centers, X_sample, k)

                        gap = calculate_gap(total_l1_distance_heuristic, total_l1_distance_milp)
                        result_entry.update({
                            'milp_l1_distance': total_l1_distance_milp,
                            'milp_time': milp_time,
                            'gap (%)': gap
                        })

                results.append(result_entry)
    
    return pd.DataFrame(results)


# Run the experiments
k_list = [2,3,4]  # Number of clusters
n_list = [15, 20]  # Number of points
results_df = run_experiments_with_milp(X, k_list, n_list, milp_enabled=True, time_limit=600)
# Print the results with detailed comparison logs
print(results_df)



# from IPython.display import display
# display(results_df)



import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Suppress the specific FutureWarning from seaborn
warnings.filterwarnings("ignore", category=FutureWarning, module='seaborn')

def plot_total_l1_distance(results_df):
    # Replace inf values with NaN in the DataFrame
    results_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Filter out rows where MILP was not run
    results_df = results_df.dropna(subset=['milp_l1_distance'])

    # Debugging: Check if the filtered DataFrame is empty
    print("Filtered DataFrame:")
    print(results_df.head())
    
    # If DataFrame is empty after filtering, stop here
    if results_df.empty:
        print("No data available for plotting after filtering.")
        return
    
    # Plot Total L1 Distance Comparison
    plt.figure(figsize=(12, 6))
    
    # Plot for heuristic (K-medians) results
    sns.lineplot(data=results_df, x='n', y='total_l1_distance', hue='k', marker='o', label='Heuristic (K-medians)')
    
    # Plot for MILP results
    sns.lineplot(data=results_df, x='n', y='milp_l1_distance', hue='k', linestyle='--', marker='x', label='MILP')
    
    # Title and labels
    plt.title('Total L1 Distance Comparison: Heuristic vs. MILP')
    plt.xlabel('Number of Points (n)')
    plt.ylabel('Total L1 Distance')
    plt.legend(title='k value')
    plt.grid(True)

    # Ensure plot shows
    plt.show()



def plot_time_taken(results_df):
    # Replace inf values with NaN in the DataFrame
    results_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Filter out rows where MILP was not run
    results_df = results_df.dropna(subset=['milp_time'])

    # Plot Time Taken Comparison
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=results_df, x='n', y='time_taken', hue='k', marker='o', label='Heuristic Time')
    sns.lineplot(data=results_df, x='n', y='milp_time', hue='k', linestyle='--', marker='x', label='MILP Time')
    plt.title('Computation Time Comparison: Heuristic vs. MILP')
    plt.xlabel('Number of Points (n)')
    plt.ylabel('Time Taken (seconds)')
    plt.legend(title='k value')
    plt.grid(True)
    plt.show()

# Assuming `results_df` is the DataFrame obtained from the run_experiments_with_milp function
plot_total_l1_distance(results_df)
plot_time_taken(results_df)






