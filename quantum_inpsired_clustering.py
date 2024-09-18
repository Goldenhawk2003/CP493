import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Quantum state generation function for 10 qubits
def random_state(num_qubits):
    dim = 2**num_qubits
    state = np.random.rand(dim) + 1j * np.random.rand(dim)
    state /= np.linalg.norm(state)
    return state

# Clustering functions
def encode_data_to_qubits(data, num_qubits):
    norm_data = data / np.linalg.norm(data)
    encoded_data = np.array([[np.real(d), np.imag(d)] for d in norm_data])
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
    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def circuit(x):
        qml.RX(x[0], wires=0)
        qml.RY(x[1], wires=0)
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

    for _ in range(max_iters):
        labels = quantum_assign_clusters(data, centroids)
        new_centroids = quantum_update_centroids(data, labels, k)
        centroid_shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids
        if np.all(previous_labels == labels) or centroid_shift < tol:
            break
        previous_labels = labels
    
    return centroids, labels

# Generate random quantum state for 10 qubits
num_qubits = 3
state = random_state(num_qubits)

data = encode_data_to_qubits(np.array(state.tolist()), num_qubits)

# Calculate probabilities (squared magnitudes of the state vector)
probabilities = np.abs(state) ** 2

# Apply quantum k-means clustering
k = 3 # Number of clusters
centroids, labels = quantum_kmeans(data, k)

# Convert data to spherical coordinates
def to_spherical(coords, probs):
    x, y = coords[:, 0], coords[:, 1]
    r = np.sqrt(x**2 + y**2 + probs)
    theta = np.arccos(np.clip(probs / r, -1.0, 1.0))  # Clip values to avoid NaNs
    phi = np.arctan2(y, x)
    return r, theta, phi

# Calculate spherical coordinates for data and centroids
spherical_data = np.column_stack(to_spherical(data, probabilities))
centroid_probs = np.array([np.mean(probabilities[labels == i]) for i in range(k)])
spherical_centroids = np.column_stack(to_spherical(centroids, centroid_probs))

# Plot the clusters in a 3D spherical plot
def plot_clusters_spherical(data, labels, centroids):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for i in range(len(centroids)):
        points = data[labels == i]
        ax.scatter(points[:, 1], points[:, 2], points[:, 0], label=f'Cluster {i}')
    ax.scatter(centroids[:, 1], centroids[:, 2], centroids[:, 0], color='black', marker='x', s=100, label='Centroids')  # s=100 sets the size of centroid markers

    ax.set_title('Quantum State Clustering (Spherical)')
    ax.set_xlabel('Theta')
    ax.set_ylabel('Phi')
    ax.set_zlabel('Radius')
    plt.legend()
    plt.show()

plot_clusters_spherical(spherical_data, labels, spherical_centroids)



# Fidelity is the overlap
# If fidelity equals 1 then they are identical
# If fidelity is 0 then they are orhtogonal (opposite)

"""The quantum state vector for 10 qubits has 2**10 =1024 components.
Each component is a complex number representing the amplitude of the corresponding basis state.
For easier visualization and to keep the plot manageable,I am gonna select 100 random components and plot them"""

#The dimensionality of the state vector is 2**10 = 1024, meaning the quantum state is represented by a complex vector of length 1024.
#The real and imaginary parts of each component of the quantum state vector are encoded into a 2D array. This results in 1024 rows, each with two columns (real and imaginary parts).