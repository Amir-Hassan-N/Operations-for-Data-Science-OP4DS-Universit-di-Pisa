
import numpy as np
import pandas as pd
import time


# ****************** Load Dataset *************************#
train = pd.read_csv('C:/Users/hp/OneDrive/Desktop/OP4DS/new/iris.csv', skipinitialspace=True)

numeric_col = train.select_dtypes(include=['float64', 'int64'])
X = numeric_col.to_numpy()


# In[4]:
X.shape
# print(X_scaled[:500]) 
# X=X[:,:1]



#  ******************** K-Median *************************#
def k_medians(X, k, max_iter=100, tol=1e-4, random_seed=None, verbose=True):
    """
    Perform K-Medians clustering using Manhattan (L1) distance.
    
    Parameters:
        X (numpy.ndarray): The dataset where rows are data points and columns are features.
        k (int): The number of clusters.
        max_iter (int): Maximum number of iterations to perform.
        tol (float): Convergence tolerance.
        random_seed (int, optional): Random seed for reproducibility.
        verbose (bool): Whether to print convergence information.
    
    Returns:
        centroids (numpy.ndarray): The coordinates of the centroids.
        labels (numpy.ndarray): The cluster assignments for each point.
        total_l1_distance (float): Total L1 distance between points and their assigned centroids.
    """
    # Ensure k is not larger than the number of points
    if k > len(X):
        raise ValueError("k cannot be greater than the number of data points.")
    
    # Set random seed for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)

    # Randomly initialize centroids from the data points
    initial_indices = np.random.choice(X.shape[0], k, replace=False)
    centroids = X[initial_indices]

    for iteration in range(max_iter):
        # Calculate Manhattan (L1) distance between each point and the centroids
        distances = np.abs(X[:, np.newaxis] - centroids).sum(axis=2)
        
        # Assign each point to the nearest centroid
        labels = np.argmin(distances, axis=1)
        
        # Update centroids as the median of points assigned to each cluster
        new_centroids = np.array([np.median(X[labels == i], axis=0) if np.sum(labels == i) > 0 else centroids[i] for i in range(k)])
        
        # Check for convergence by comparing new and old centroids
        centroid_shift = np.abs(new_centroids - centroids).sum()
        if verbose:
            print(f"Iteration {iteration + 1}: Centroid shift = {centroid_shift:.4f}")
        
        if centroid_shift < tol:
            if verbose:
                print(f"Converged after {iteration + 1} iterations.")
            break
        
        centroids = new_centroids
    
    # Compute total L1 distance (sum of distances between each point and its centroid)
    total_l1_distance = np.sum([np.sum(np.abs(X[labels == i] - centroids[i])) for i in range(k)])

    if verbose:
        print(f"Total L1 distance after convergence: {total_l1_distance:.4f}")
    
    return centroids, labels, total_l1_distance



import time
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor

# ******************* Experiments *************************#
def run_experiments(X, k_list, n_list, random_seed=46, verbose=True, use_parallel=False):
    """
    Run experiments using the K-medians heuristic on different subsets of the dataset.
    
    Parameters:
        X (numpy.ndarray): The dataset where rows are data points and columns are features.
        k_list (list): List of k values (number of clusters) to test.
        n_list (list): List of n values (number of points) to sample from the dataset.
        random_seed (int, optional): Random seed for reproducibility. Default is 46.
        verbose (bool, optional): Whether to print progress information. Default is True.
        use_parallel (bool, optional): Whether to run experiments in parallel using multiple processors.
    
    Returns:
        pandas.DataFrame: A DataFrame with results for each combination of n and k.
    """
    results = []
    np.random.seed(random_seed)

    # Cache the sampled data for each n to avoid redundant sampling
    sampled_data = {n: X[np.random.choice(X.shape[0], n, replace=False)] if len(X) > n else X for n in n_list}

    if verbose:
        print(f"Sampled data prepared for n values: {n_list}")
    
    def run_single_experiment(n, k, X_sample):
        if k > n:
            return None  # Invalid configuration, skip

        start_time = time.time()
        try:
            centroids, labels, total_l1_distance = k_medians(X_sample, k, random_seed=random_seed, verbose=False)
            heuristic_time = time.time() - start_time
            return {
                'n': n,
                'k': k,
                'method': 'K-medians',
                'total_l1_distance': total_l1_distance,
                'time_taken': heuristic_time,
                'random_seed': random_seed
            }
        except Exception as e:
            if verbose:
                print(f"Error for n={n}, k={k}: {str(e)}")
            return None
    
    if use_parallel:
        # Use parallel processing
        with ProcessPoolExecutor() as executor:
            futures = []
            for n in n_list:
                X_sample = sampled_data[n]
                for k in k_list:
                    futures.append(executor.submit(run_single_experiment, n, k, X_sample))
            
            for future in futures:
                result = future.result()
                if result:
                    results.append(result)
    else:
        # Run experiments sequentially
        for n in n_list:
            X_sample = sampled_data[n]
            if verbose:
                print(f"Running experiments for n={n} points")

            for k in k_list:
                if verbose:
                    print(f"Running K-medians for n={n}, k={k}...")
                
                result = run_single_experiment(n, k, X_sample)
                if result:
                    results.append(result)
    
    # Convert the results into a DataFrame
    results_df = pd.DataFrame(results)
    
    if verbose:
        print("Experiment completed. Summary of results:")
#         print(results_df)
    
    return results_df


# Example usage
k_list = [2, 3, 4]  # Example values for k
n_list = [15, 20]

# Run experiments
results_df = run_experiments(X, k_list, n_list)


# Display the results
print(results_df)



import matplotlib.pyplot as plt

# 88********************** Plotting function ****************************#
def plot_results(results_df):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot total L1 distance
    for k in results_df['k'].unique():
        subset = results_df[results_df['k'] == k]
        ax1.plot(subset['n'], subset['total_l1_distance'], label=f'k={k}', marker='o')

    ax1.set_xlabel('Number of Points (n)')
    ax1.set_ylabel('Total L1 Distance')
    ax1.set_title('K-medians Heuristic: Total L1 Distance vs Number of Points')
    ax1.legend()

    # Create a second y-axis for the time taken
    ax2 = ax1.twinx()
    for k in results_df['k'].unique():
        subset = results_df[results_df['k'] == k]
        ax2.plot(subset['n'], subset['time_taken'], label=f'k={k} (time)', linestyle='--', marker='x', color='gray')

    ax2.set_ylabel('Time Taken (seconds)')
    ax2.legend(loc='upper right')

    plt.show()

# Plot the results
plot_results(results_df)


# ******************** Ploting *********************** #
import matplotlib.pyplot as plt
import seaborn as sns

# Plot the results: Time Taken vs Number of Data Points (n)
def plot_results(results_df):
    plt.figure(figsize=(14, 6))

    # Plot 1: Time Taken vs Number of Data Points (n) for different k
    plt.subplot(1, 2, 1)
    sns.lineplot(data=results_df, x='n', y='time_taken', hue='k', marker='o')
    plt.title('Time Taken by K-medians vs Number of Data Points')
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



