import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
from mpl_toolkits.mplot3d import Axes3D

# Quantum device
num_qubits = 2
dev = qml.device("default.qubit", wires=num_qubits)

# Quantum circuit to create entangled states
@qml.qnode(dev)
def entangled_circuit():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.state()

# Generate entangled qubits
def generate_entangled_qubits(num_samples):
    states = []
    for _ in range(num_samples):
        state = entangled_circuit()
        states.append(state)
    return np.array(states)

# Clustering functions
def encode_data_to_qubits(data):
    norm_data = data / np.linalg.norm(data)
    encoded_data = np.column_stack([np.real(norm_data), np.imag(norm_data)])
    return encoded_data

def quantum_initialize_centroids(data, k):
    np.random.seed(42)
    centroids = [data[np.random.choice(len(data))]]
    for _ in range(1, k):
        distances = [min([quantum_distance(d, c) for c in centroids]) for d in data]
        probabilities = distances / np.sum(distances)
        cumulative_probabilities = np.cumsum(probabilities)
        r = np.random.rand()
        for i, p in enumerate(cumulative_probabilities):
            if r < p:
                centroids.append(data[i])
                break
    return np.array(centroids)

def quantum_distance(a, b):
    def pad_to_length(x, length):
        return np.pad(x, (0, length - len(x)), 'constant')

    @qml.qnode(dev)
    def circuit(x):
        qml.AmplitudeEmbedding(pad_to_length(x, 4), wires=[0, 1], normalize=True)
        return qml.state()

    state_a = circuit(a)
    state_b = circuit(b)
    fidelity = np.abs(np.dot(np.conj(state_a), state_b))**2
    return 1 - fidelity

def quantum_assign_clusters(data, centroids):
    labels = np.zeros(len(data))
    for i, d in enumerate(data):
        distances = [quantum_distance(d, c) for c in centroids]
        labels[i] = np.argmin(distances)
    return labels

def quantum_update_centroids(data, labels, k):
    new_centroids = np.zeros((k, data.shape[1]))
    for i in range(k):
        points = data[labels == i]
        if len(points) > 0:
            new_centroids[i] = np.mean(points, axis=0)
        else:
            new_centroids[i] = data[np.random.choice(len(data))]
    return new_centroids

def quantum_kmeans(data, k, max_iters=100, tol=1e-4):
    centroids = quantum_initialize_centroids(data, k)
    previous_labels = np.zeros(len(data))

    for iteration in range(max_iters):
        labels = quantum_assign_clusters(data, centroids)
        new_centroids = quantum_update_centroids(data, labels, k)
        centroid_shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids
        print(f"Iteration {iteration}: centroid shift = {centroid_shift}")
        if np.all(previous_labels == labels) or centroid_shift < tol:
            break
        previous_labels = labels
    
    return centroids, labels

# Generate entangled qubits for clustering
num_samples = 500  # Increase the number of samples
entangled_states = generate_entangled_qubits(num_samples)
data = encode_data_to_qubits(entangled_states.flatten())

# Calculate probabilities (squared magnitudes of the selected state vector)
probabilities = np.abs(entangled_states.flatten()) ** 2

# Plot initial data distribution
def plot_initial_data(data, title):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    r, theta, phi = to_spherical(data, probabilities)
    ax.scatter(phi, theta, r, color='blue', label='Data')
    ax.set_title(title)
    ax.set_xlabel('Phi')
    ax.set_ylabel('Theta')
    ax.set_zlabel('Radius')
    plt.legend()
    plt.show()

# Convert data to spherical coordinates
def to_spherical(coords, probs):
    x, y = coords[:, 0], coords[:, 1]
    r = np.sqrt(x**2 + y**2 + probs)
    r[r == 0] = np.finfo(float).eps  # Avoid division by zero
    theta = np.arccos(np.clip(probs / r, -1.0, 1.0))  # Clip values to avoid NaNs
    phi = np.arctan2(y, x)
    return r, theta, phi

plot_initial_data(data, "Initial Data Distribution")

# Apply quantum k-means clustering
k = 3  # Number of clusters
centroids, labels = quantum_kmeans(data, k)

# Calculate spherical coordinates for data and centroids
spherical_data = np.column_stack(to_spherical(data, probabilities))

# Ensure valid centroid probabilities
centroid_probs = []
for i in range(k):
    cluster_probs = probabilities[labels == i]
    if len(cluster_probs) > 0:
        centroid_probs.append(np.mean(cluster_probs))
    else:
        centroid_probs.append(np.finfo(float).eps)  # Assign small value to avoid NaN

centroid_probs = np.array(centroid_probs)
spherical_centroids = np.column_stack(to_spherical(centroids, centroid_probs))

# Verify spherical data
print("Spherical data sample:", spherical_data[:5])
print("Spherical centroids:", spherical_centroids)

# Plot the clusters in a 3D spherical plot
def plot_clusters_spherical(data, labels, centroids):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(len(centroids)):
        points = data[labels == i]
        ax.scatter(points[:, 2], points[:, 1], points[:, 0], label=f'Cluster {i}')
    ax.scatter(centroids[:, 2], centroids[:, 1], centroids[:, 0], color='black', marker='x', s=100, label='Centroids')

    ax.set_title('Quantum State Clustering (Spherical)')
    ax.set_xlabel('Phi')
    ax.set_ylabel('Theta')
    ax.set_zlabel('Radius')
    plt.legend()
    plt.show()

plot_clusters_spherical(spherical_data, labels, spherical_centroids)