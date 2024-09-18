import numpy as np

def euclidean_distance_complex(z1, z2):
    return np.abs(z1 - z2)

def silhouette_score_complex(data, labels, k):
    n = len(data)
    a = np.zeros(n)
    b = np.zeros(n)
    
    for i in range(n):
        same_cluster = data[labels == labels[i]]
        other_clusters = [data[labels == l] for l in range(k) if l != labels[i]]
        
        # Calculate a(i)
        if len(same_cluster) > 1:
            a[i] = np.mean([euclidean_distance_complex(data[i], p) for p in same_cluster if not np.array_equal(p, data[i])])
        else:
            a[i] = 0
        
        # Calculate b(i)
        b[i] = np.min([np.mean([euclidean_distance_complex(data[i], p) for p in cluster]) for cluster in other_clusters])
    
    s = (b - a) / np.maximum(a, b)
    print(np.mean(s))
    return np.mean(s)

def davies_bouldin_index(data, centroids, assignments, k):
    n_clusters = len(centroids)
    cluster_scatters = np.zeros(n_clusters)
    cluster_distances = np.zeros((n_clusters, n_clusters))
    
    for i in range(n_clusters):
        cluster_points = data[assignments == i]
        if len(cluster_points) > 0:
            cluster_scatters[i] = np.mean([np.linalg.norm(point - centroids[i]) for point in cluster_points])
    
    for i in range(n_clusters):
        for j in range(n_clusters):
            if i != j:
                cluster_distances[i, j] = np.linalg.norm(centroids[i] - centroids[j])
    
    db_indexes = np.zeros(n_clusters)
    
    for i in range(n_clusters):
        max_ratio = 0
        for j in range(n_clusters):
            if i != j:
                ratio = (cluster_scatters[i] + cluster_scatters[j]) / cluster_distances[i, j]
                if ratio > max_ratio:
                    max_ratio = ratio
        db_indexes[i] = max_ratio
    
    dbi = np.mean(db_indexes)
    return dbi


