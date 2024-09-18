import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from evaluate import silhouette_score_complex, davies_bouldin_index

num_qubits = 3  # 2 data qubits + 1 ancilla qubit for SWAP test
dev = qml.device("default.qubit", wires=num_qubits)

@qml.qnode(dev)
def swap_test(state1, state2):
    qml.Hadamard(wires=0)
    qml.AngleEmbedding(state1, wires=[1, 2], rotation='Y')
    qml.AngleEmbedding(state2, wires=[1, 2], rotation='Y')
    qml.CSWAP(wires=[0, 1, 2])
    qml.Hadamard(wires=0)
    return qml.expval(qml.PauliZ(0))

def fidelity(state1, state2):
    return (swap_test(state1, state2) + 1) / 2

def distance(a, b):
    return 1 - fidelity(a, b)

def kmeans_plus_plus_initialization(data, k):
    np.random.seed(42)
    centroids = [data[np.random.choice(len(data))]]
    
    for _ in range(1, k):
        distances = np.array([min([distance(x, c) for c in centroids]) for x in data])
        probabilities = distances / distances.sum()
        cumulative_probabilities = np.cumsum(probabilities)
        r = np.random.rand()
        for i, p in enumerate(cumulative_probabilities):
            if r < p and not any(np.allclose(data[i], c) for c in centroids):
                centroids.append(data[i])
                break
    
    return np.array(centroids)

def recalculate_centroids(data, assignments, k):
    new_centroids = []
    for i in range(k):
        cluster_data = data[assignments == i]
        if len(cluster_data) > 0:
            new_centroid = np.mean(cluster_data, axis=0)
        else:
            new_centroid = None
        new_centroids.append(new_centroid)
    return new_centroids

def assign_clusters(data, centroids):
    distances = np.array([[distance(state, centroid) for centroid in centroids] for state in data])
    return np.argmin(distances, axis=1)

def optimize_k_means(data, k, max_iterations=100, n_runs=10):
    best_assignments = None
    best_centroids = None
    best_silhouette = -1
    best_dbi = float('inf')

    for run in range(n_runs):
        centroids = kmeans_plus_plus_initialization(data, k)
        for iteration in range(max_iterations):
            assignments = assign_clusters(data, centroids)
            new_centroids = recalculate_centroids(data, assignments, k)

            # Check if any centroid is None (empty cluster case)
            for idx, centroid in enumerate(new_centroids):
                if centroid is None:
                    # Reassign points to the closest non-empty centroids
                    valid_centroids = [c for c in new_centroids if c is not None]
                    for i in range(len(data)):
                        if assignments[i] == idx:
                            distances = np.array([distance(data[i], vc) for vc in valid_centroids])
                            new_assignment = np.argmin(distances)
                            assignments[i] = new_centroids.index(valid_centroids[new_assignment])
                    new_centroids = recalculate_centroids(data, assignments, k)

            if np.all([centroid is not None for centroid in new_centroids]):
                centroids = np.array(new_centroids)
                if np.allclose(centroids, new_centroids):
                    break

        silhouette_avg = silhouette_score_complex(data, assignments, k)
        dbi = davies_bouldin_index(data, centroids, assignments, k)

        if silhouette_avg > best_silhouette or (silhouette_avg == best_silhouette and dbi < best_dbi):
            best_assignments = assignments
            best_centroids = centroids
            best_silhouette = silhouette_avg
            best_dbi = dbi

    return best_assignments, best_centroids, best_silhouette, best_dbi

# Example data
data = np.array([np.random.rand(2) for _ in range(10)])  # Example data with 2 features

k = 2 # Number of clusters
assignments, centroids, silhouette_avg, dbi = optimize_k_means(data, k, max_iterations=100, n_runs=10)

print("Cluster assignments:", assignments)
print(f"Silhouette Score: {silhouette_avg}")
print(f"Davies-Bouldin Index: {dbi}")

# Plotting results
plt.figure(figsize=(10, 6))
colors = plt.get_cmap("tab10")

for i in range(k):
    cluster_data = data[assignments == i]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], color=colors(i), alpha=0.7, label=f'Cluster {i} Data')

for i, centroid in enumerate(centroids):
    plt.scatter(centroid[0], centroid[1], color='k', marker="x", s=100, label=f'Centroid {i}', alpha=0.7)

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Quantum K-Means Clustering')
plt.legend()
plt.grid(True)
plt.show()