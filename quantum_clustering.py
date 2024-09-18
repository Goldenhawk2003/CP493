import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

num_qubits = 3  # Increased to 3 to accommodate the ancilla qubit for the SWAP test

dev = qml.device("default.qubit", wires=num_qubits)

@qml.qnode(dev)
def embed(x):
    qml.AngleEmbedding(features=[x, 0], wires=[1, 2], rotation="Z")
    return qml.state()

data = np.array([0.1, 0.2, 0.3, 0.4, 2, 8, 20, 200, 2000, 10000, 10200, 10100])

states_list = []
for x in data:
    state = embed(x)
    states_list.append(state)

states_array = np.array(states_list)

k = 3

def distance(a, b):
    fidelity = np.abs(np.dot(np.conj(a), b))**2
    return 1 - fidelity

def kmeans_plus_plus_initialization(data, k):
    np.random.seed(42)  # For reproducibility
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

centroids = kmeans_plus_plus_initialization(states_array, k)
print("Initial centroids:", centroids)

def compute_euclidean_distance(x1, x2):
    return np.linalg.norm(np.concatenate([np.real(x1), np.imag(x1)]) - np.concatenate([np.real(x2), np.imag(x2)]))

distances = np.zeros((len(states_array), k))
for i, state in enumerate(states_array):
    for j, centroid in enumerate(centroids):
        dist = compute_euclidean_distance(state, centroid)
        distances[i, j] = dist
        print(f"Distance from data point {i} to centroid {j}: {dist}")

assignments = np.argmin(distances, axis=1)
print("Cluster assignments:", assignments)

# Plot the real and imaginary parts of the states along with the centroids
plt.figure(figsize=(10, 6))

colors = plt.get_cmap("tab10")
markers = ['o', 's', 'D', '^', 'v', 'p', '*']

for i in range(k):
    cluster_data = states_array[assignments == i]
    for state in cluster_data:
        plt.scatter(np.real(state), np.imag(state), color=colors(i / k), alpha=0.7, label=f'Cluster {i} Data' if i == 0 else "")

for i, centroid in enumerate(centroids):
    plt.scatter(np.real(centroid), np.imag(centroid), color='k', marker=markers[i % len(markers)], s=100, label=f'Centroid {i}', alpha=0.7)

plt.xlabel('Real part')
plt.ylabel('Imaginary part')
plt.title('Scatter Plot of Quantum State Components with Centroids')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.grid(True)
plt.show()
#https://arxiv.org/pdf/2112.08506