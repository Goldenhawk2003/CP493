import numpy as np
import matplotlib.pyplot as plt

def initialize_centroids_kmeans_plus_plus(data, k):
    """Initialize centroids using the k-means++ algorithm."""
    # Set the random seed for reproducibility
    np.random.seed(42)
    # Randomly select the first centroid from the data points
    centroids = [data[np.random.choice(data.shape[0])]]
    for _ in range(1, k):
        # Compute the distance between each data point and the nearest centroid
        distance = np.min(np.linalg.norm(data[:, np.newaxis] - centroids, axis=2), axis=1)
        # Calculate the probabilities for selecting the next centroid
        probabilities = distance / np.sum(distance)
        # Compute the cumulative probabilities
        cumulative_probabilities = np.cumsum(probabilities)
        # Generate a random number
        r = np.random.rand()
        # Select the next centroid based on the cumulative probabilities
        for j, p in enumerate(cumulative_probabilities):
            if r < p:
                centroids.append(data[j])
                break
    return np.array(centroids)

def assign_clusters(data, centroids):
    """Assign each data point to the nearest centroid."""
    # Compute the distance between each data point and each centroid
    distance = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
    # Assign each data point to the cluster with the nearest centroid
    return np.argmin(distance, axis=1)

def update_centroids(data, labels, k):
    """Calculate new centroids as the mean of the points in each cluster."""
    # Calculate the mean of the points in each cluster to update centroids
    centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
    for i in range(k):
        # Handle empty clusters by reinitializing the centroid randomly
        if np.isnan(centroids[i]).any():
            centroids[i] = data[np.random.choice(data.shape[0])]
    return centroids

def kmeans(data, k, max_iters=100, tol=1e-4):
    """K-means clustering algorithm."""
    # Initialize centroids using k-means++
    centroids = initialize_centroids_kmeans_plus_plus(data, k)
    # Initialize an array to store previous cluster assignments
    previous_labels = np.zeros(data.shape[0])

    for _ in range(max_iters):
        # Assign data points to the nearest centroid
        labels = assign_clusters(data, centroids)
        # Update centroids based on the current cluster assignments
        new_centroids = update_centroids(data, labels, k)
        # Calculate the shift in centroid positions
        centroid_shift = np.linalg.norm(new_centroids - centroids)
        # Update centroids for the next iteration
        centroids = new_centroids
        # Check for convergence based on label changes and centroid shifts
        if np.all(previous_labels == labels) or centroid_shift < tol:
            break
        # Update previous labels for the next iteration
        previous_labels = labels
    return centroids, labels



if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    data = np.random.rand(1000, 2)

    # Apply k-means clustering
    k = 3
    centroids, labels = kmeans(data, k)

    # Plot the clusters
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x')
    plt.title("K-Means Clustering")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

"""

    - list all machine learning algorithms that have been done in quantum computing
    - try recreating the circuit on d-wave
    - find a set of entangled qubits for the quantum version and organize them through clusters

"""