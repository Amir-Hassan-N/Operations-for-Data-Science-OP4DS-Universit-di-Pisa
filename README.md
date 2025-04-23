# 📊 Constrained Clustering Optimization Using Heuristic and Exact Methods

## Optimizing Centroid Assignment with L1 Norm (Manhattan Distance)
### A project for Operations for Data Science (OP4DS) – Università di Pisa

📌 Overview
This project explores the Constrained Clustering Problem under the L1 norm (Manhattan distance) using both heuristic (k-medians) and exact (Mixed Integer Linear Programming - MILP) approaches. The goal is to identify k centroids and assign n data points to them in a way that minimizes the total sum of L1 distances.

Two algorithms were implemented:

A1 - Heuristic approach: Adapted k-medians algorithm for L1 norm

A2 - Exact approach: MILP formulation solved using CBC (COIN-OR Branch and Cut) optimizer

🧠 Problem Statement
Given a dataset with n points in d dimensions, the objective is to:

Find k centroids

Assign each data point to exactly one centroid

Minimize the sum of L1 distances (|x - c|) between points and their respective centroids

🧪 Methods
🔹 Heuristic Approach (A1) – Modified K-Medians
Steps:

Randomly initialize centroids

Assign each point to the closest centroid (L1 distance)

Update centroids using median of assigned points

Repeat until convergence

Strengths: Fast, scalable

Limitations: Sensitive to initial centers, may reach local optima

🔹 Exact Approach (A2) – MILP Formulation
Binary variables for assignments (yᵢⱼ ∈ {0,1})

Objective: Minimize total L1 distance

Constraints:

Each point must be assigned to exactly one centroid

L1 distances computed using auxiliary variables

Centroids constrained within bounding box of input data

Solver: CBC (open-source MILP solver)

Strengths: Guarantees optimal solution

Limitations: Computationally expensive for large n or k

📁 Dataset
Dataset Used: Iris Dataset (3 classes, 4 numerical features)

Used to test performance of both methods on a real-world, balanced dataset

Synthetic datasets also used for scalability testing

📈 Results Summary

n	k	Method	Total L1 Distance	Time Taken (s)	GAP (%)
15	2	K-Medians	2.9	0.019	7.4%
15	2	CBC MILP	2.7	0.73	–
20	4	K-Medians	2.0	0.0013	25.0%
20	4	CBC MILP	1.6	432.31	–
Heuristic is up to 1000x faster, with slightly lower accuracy

MILP produces optimal clusters but struggles with scalability

📊 Visualizations
Line plots: Total L1 distance vs. k

Bar charts: Time comparison between methods

GAP analysis: Percent deviation of heuristic from MILP results

Graphs confirm that clustering improves with higher k, but computation time for MILP skyrockets

⚙️ Technologies Used
Languages: Python

Optimization Solver: CBC (COIN-OR)

Libraries: NumPy, Matplotlib, PuLP, PyCBC, Scikit-learn

✅ Key Takeaways
The heuristic k-medians approach is well-suited for large datasets and fast execution

MILP guarantees optimal clustering, ideal when accuracy outweighs compute constraints

For real-time applications or limited compute resources, heuristics offer a strong trade-off

🛣️ Future Enhancements
Apply MILP to larger datasets using problem decomposition

Integrate k-means++ initialization to improve heuristic accuracy

Experiment with other solvers (e.g., Gurobi, CPLEX) for faster exact optimization

Develop a hybrid model: heuristic to initialize MILP variables

👥 Authors
Amir Hassan

📫 Contact
📧 amirhassanunipi29@gmail.com
