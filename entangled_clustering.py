import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from evaluate import silhouette_score_complex
from evaluate import davies_bouldin_index
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import rand_score
# Define the number of qubits
num_qubits = 2

# Define the quantum device
dev = qml.device("default.qubit", wires=num_qubits)

# Define the quantum circuit with angle embedding
@qml.qnode(dev)
def normalize(x1, x2):
    qml.AngleEmbedding(features=[x1], wires=[0], rotation='Z')
    qml.AngleEmbedding(features=[x2], wires=[1], rotation='Z')
    qml.Hadamard(0)
    qml.Hadamard(1)
    qml.CNOT(wires=[0, 1])
    return qml.state()

# Example classical data
iris = datasets.load_iris()
data = iris.data 
ground_truth = iris.target 

scaler = MinMaxScaler()
data = scaler.fit_transform(data.reshape(-1, 1)).flatten()

# Split data into two halves
data1 = data[:len(data)//2]
data2 = data[len(data)//2:]

# Apply the normalize function to the data
states_list = []
for x1, x2 in zip(data1, data2):
    state = normalize([x1], [x2])
    states_list.append(state)

# Convert states_list to a numpy array for easy manipulation
states_array = np.array(states_list)

# Define the number of clusters
k = 3  # Changed k to 2

# k-means++ initialization
def kmeans_plus_plus_initialization(data, k):
    np.random.seed(42)  # For reproducibility
    centroids = []
    # Randomly select the first centroid
    centroids.append(data[np.random.choice(len(data))])
    
    for _ in range(1, k):
        distances = np.array([min([np.linalg.norm(x - c) for c in centroids]) for x in data])
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
    for _ in range(max_iter):
        # Measure distances between each data point and all centroids
        distances = np.zeros((len(data), k))
        for i, state in enumerate(data):
            for j, centroid in enumerate(centroids):
                distances[i, j] = np.linalg.norm(state - centroid)
        
        # Assign each data point to the nearest centroid
        assignments = np.argmin(distances, axis=1)
        
        # Update the centroids
        new_centroids = update_centroids(data, assignments, k)
        
        # Check for convergence (if centroids do not change)
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return centroids, assignments

centroids, assignments = quantum_kmeans(states_array, k)

silhouette_avg = silhouette_score_complex(states_array, assignments, k)
print(f"Silhouette Score: {silhouette_avg}")

dbi = davies_bouldin_index(states_array, centroids, assignments, k)
print(f"Davies-Bouldin Index: {dbi}")

#ri = rand_score(ground_truth, assignments)
#print(f"Rand Index: {ri}")

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