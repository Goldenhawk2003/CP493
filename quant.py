import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from evaluate import silhouette_score_complex
from evaluate import davies_bouldin_index
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler

# Define the number of qubits
num_qubits = 3

# Define the quantum device with 3 wires for the swap test
dev = qml.device("default.qubit", wires=num_qubits)

# Define the quantum circuit with angle embedding
@qml.qnode(dev)
def normalize(x):
    qml.AngleEmbedding(features=[x], wires=[0], rotation='Z')
    qml.Hadamard(0)
    return qml.state()

@qml.qnode(dev)
def swap_test(state1, state2):
    # Initialize the states on the appropriate wires using QubitStateVector
    qml.QubitStateVector(state1, wires=[0, 1, 2])  # assign to all wires
    qml.QubitStateVector(state2, wires=[0, 1, 2])  # assign to all wires
    qml.Hadamard(wires=0)
    qml.CSWAP(wires=[0, 1, 2])
    qml.Hadamard(wires=0)
    return qml.probs(wires=0)

def swap_test_distance(state1, state2):
    state1 = state1 / np.linalg.norm(state1)  # Normalize before using in swap test
    state2 = state2 / np.linalg.norm(state2)  # Normalize before using in swap test
    probs = swap_test(state1, state2)
    return 1 - 2 * probs[0]  # Using the probability of measuring 0 as the distance

# Load the Iris dataset
iris = datasets.load_iris()
data = iris.data[:, 0]  # Using only one feature for simplicity
ground_truth = iris.target

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler()
data = scaler.fit_transform(data.reshape(-1, 1)).flatten()

# Apply the normalize function to the data
states_list = []
for x in data:
    state = normalize(x)
    state = state / np.linalg.norm(state)  # Ensure the state vector is normalized
    if len(state) != 8:
        state = np.pad(state, (0, 8 - len(state)), 'constant')
        state = state / np.linalg.norm(state)  # Normalize again after padding
    print(f'State shape: {state.shape}, Norm: {np.sum(np.abs(state)**2)}')  # Debug print statement
    states_list.append(state)

# Convert states_list to a numpy array for easy manipulation
states_array = np.array(states_list)

# Ensure all states have the correct shape and are normalized
for i, state in enumerate(states_array):
    if state.shape != (8,):
        print(f'Error in state shape at index {i}: {state.shape}')  # Debug print statement
    if not np.isclose(np.sum(np.abs(state)**2), 1.0):
        print(f"Error in normalization at index {i}: Sum of amplitudes-squared is {np.sum(np.abs(state)**2)}")

# Define the number of clusters
k = int(input("Number of clusters: "))

# k-means++ initialization
def kmeans_plus_plus_initialization(data, k):
    np.random.seed(42)  # For reproducibility
    centroids = []
    # Randomly select the first centroid
    centroids.append(data[np.random.choice(len(data))])
    
    for _ in range(1, k):
        distances = np.array([min([swap_test_distance(x, c) for c in centroids]) for x in data])
        probabilities = distances / distances.sum()
        cumulative_probabilities = np.cumsum(probabilities)
        r = np.random.rand()
        for i, p in enumerate(cumulative_probabilities):
            if r < p:
                centroids.append(data[i])
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
def quantum_kmeans(data, k, max_iter=100):
    centroids = kmeans_plus_plus_initialization(data, k)
    for iter_num in range(max_iter):
        print(f"Iteration {iter_num + 1}/{max_iter}")  # Debug print statement
        # Measure distances between each data point and all centroids using the swap test
        distances = np.zeros((len(data), k))
        for i, state in enumerate(data):
            for j, centroid in enumerate(centroids):
                distances[i, j] = swap_test_distance(state, centroid)
        
        # Assign each data point to the nearest centroid
        assignments = np.argmin(distances, axis=1)
        
        # Update the centroids
        new_centroids = update_centroids(data, assignments, k)
        
        # Check for convergence (if centroids do not change)
        if np.allclose(centroids, new_centroids):
            print(f"Converged after {iter_num + 1} iterations")  # Debug print statement
            break
        centroids = new_centroids
    return centroids, assignments

centroids, assignments = quantum_kmeans(states_array, k)

# Calculate the silhouette score for complex data
silhouette_avg = silhouette_score_complex(states_array, assignments, k)
print(f"Silhouette Score: {silhouette_avg}")

dbi = davies_bouldin_index(states_array, centroids, assignments, k)
print(f"Davies-Bouldin Index: {dbi}")

# Plot the real and imaginary parts of the states along with the centroids
plt.figure(figsize=(10, 6))

colors = plt.cm.get_cmap("tab10", k)

for i, state in enumerate(states_list):
    real_parts = np.real(state)
    imag_parts = np.imag(state)
    plt.scatter(real_parts, imag_parts, color=colors(assignments[i]), label=f'Data {data[i]}', alpha=0.7)

for i, centroid in enumerate(centroids):
    real_parts = np.real(centroid)
    imag_parts = np.imag(centroid)
    plt.scatter(real_parts, imag_parts, color='k', marker='x', s=100, label=f'Centroid {i}', alpha=0.7)

plt.xlabel('Real part')
plt.ylabel('Imaginary part')
plt.title('Scatter Plot of Quantum State Components with Centroids')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.grid(True)
plt.show()