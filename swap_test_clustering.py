import pennylane as qml
from pennylane import numpy as np
from math import pi
import matplotlib.pyplot as plt

# Define the number of qubits
num_qubits = 3

# Define the quantum device
dev = qml.device("default.qubit", wires=num_qubits)

@qml.qnode(dev)
def normalize(x):
    qml.AngleEmbedding(features=[x], wires=[1], rotation='Z')
    qml.Hadamard(0)
    return qml.state()

@qml.qnode(dev)
def normalize_other(x):
    qml.AngleEmbedding(features=[x], wires=[2], rotation='Z')
    qml.Hadamard(0)
    return qml.state()

# Define the quantum circuit for the swap test
@qml.qnode(dev)
def swap_test_circuit(theta1, theta2):
    # Initialize the auxiliary qubit
    qml.Hadamard(wires=0)

    # Apply the rotations for the first point
    qml.Hadamard(wires=1)
    qml.RY(theta1, wires=1)

    # Apply the rotations for the second point
    qml.Hadamard(wires=2)
    qml.RY(theta2, wires=2)

    # Apply the swap test
    qml.CSWAP(wires=[0, 1, 2])
    qml.Hadamard(wires=0)

    # Measure the auxiliary qubit
    return qml.probs(wires=0)

# Define the function to calculate the swap test distance
def swap_test_distance(theta1, theta2):
    probs = swap_test_circuit(theta1, theta2)
    return float(1 - 2 * probs[0])

data = np.array([0.3243, 21312, 21, 120123230, 0.58, 0.95, 1221, 123, 2133, 23123]) # Using only one feature for simplicity

# Apply the normalize function to the data
states_list = []
for x in data:
    state = normalize(x)
    state = state / np.linalg.norm(state)  # Ensure the state vector is normalized
    if len(state) != 8:
        state = np.pad(state, (0, 8 - len(state)), 'constant')
        state = state / np.linalg.norm(state)  # Normalize again after padding
    states_list.append(state)

states_array = np.array(states_list)

k = int(input("Number of clusters: "))

def kmeans_plus_plus_initialization(states_array, k):
    np.random.seed(42)  # For reproducibility
    centroids = []
    # Randomly select the first centroid
    centroids.append(states_array[np.random.choice(len(states_array))])
    
    for _ in range(1, k):
        distances = np.array([min([swap_test_distance(x, c) for c in centroids]) for x in states_array])
        probabilities = distances / distances.sum()
        cumulative_probabilities = np.cumsum(probabilities)
        r = np.random.rand()
        for i, p in enumerate(cumulative_probabilities):
            if r < p:
                centroids.append(states_array[min(i, len(states_array) - 1)])
                break
    
    return np.array(centroids)

centroids = kmeans_plus_plus_initialization(states_array, k)

# Function to update centroids
def update_centroids(data, assignments, k):
    new_centroids = []
    for i in range(k):
        cluster_points = data[assignments == i]
        if len(cluster_points) > 0:
            new_centroid = np.mean(cluster_points, axis=0)
            new_centroids.append(new_centroid)
        else:
            new_centroids.append(data[np.random.choice(len(data))])
    return np.array(new_centroids)

# Function to perform k-means clustering using quantum principles
def quantum_kmeans(states_array, k, max_iter=100):
    centroids = kmeans_plus_plus_initialization(states_array, k)
    for iter_num in range(max_iter):
        print(f"Iteration {iter_num + 1}/{max_iter}")
        # Measure distances between each data point and all centroids using the swap test
        distances = np.zeros((len(states_array), k))
        for i, state in enumerate(states_array):
            for j, centroid in enumerate(centroids):
                distances[i, j] = swap_test_distance(state, centroid)
        
        # Assign each data point to the nearest centroid
        assignments = np.argmin(distances, axis=1)
        
        # Update the centroids
        new_centroids = update_centroids(states_array, assignments, k)
        
        # Check for convergence (if centroids do not change)
        if np.allclose(centroids, new_centroids):
            print(f"Converged after {iter_num + 1} iterations")
            break
        centroids = new_centroids
    return centroids, assignments

centroids, assignments = quantum_kmeans(states_array, k)

plt.figure(figsize=(10, 6))

colors = plt.cm.get_cmap("tab10", k)

for i, state in enumerate(states_list):
    real_parts = np.real(state)
    imag_parts = np.imag(state)
    plt.scatter(real_parts, imag_parts, color=colors(assignments[i]), label=f'Data {data[i]}' if i == 0 else "", alpha=0.7)

for i, centroid in enumerate(centroids):
    real_parts = np.real(centroid)
    imag_parts = np.imag(centroid)
    plt.scatter(real_parts, imag_parts, color='k', marker='x', s=100, label=f'Centroid {i}', alpha=0.7)

plt.xlabel('Real part')
plt.ylabel('Imaginary part')
plt.title('Scatter Plot of Quantum State Components with Centroids')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.grid(True)
plt.show()