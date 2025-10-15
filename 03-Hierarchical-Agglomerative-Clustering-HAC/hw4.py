from csv import DictReader
import numpy as np
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt

# 0.1: Load data from CSV file
def load_data(filepath):
    with open(filepath, 'r') as file:
        rdr = DictReader(file)
        rows = list(rdr)
        
    return rows

# 0.2 Calculate Features
def calc_features(row):
    # Statistics -> Float
    x1 = float(row['Population'])
    x2 = float(row['Net migration'])
    x3 = float(row['GDP ($ per capita)'])
    x4 = float(row['Literacy (%)'])
    x5 = float(row['Phones (per 1000)'])
    x6 = float(row['Infant mortality (per 1000 births)'])
    features_array = np.array([x1, x2, x3, x4, x5, x6], dtype=np.float64)
    
    return features_array

# 0.3 HAC
def hac(features):

    n = len(features)

    cluster_indices = list(range(n))

    Z = np.zeros((n - 1, 4))

    distance_matrix = np.zeros((n, n))

    for i in range(n):

        for j in range(i + 1, n):

            distance = np.linalg.norm(features[i] - features[j])

            distance_matrix[i, j] = distance

    new_cluster_index = n

    for k in range(n - 1):

        min_distance = np.inf

        cluster1 = -1

        cluster2 = -1

        for i in range(len(cluster_indices)):

            for j in range(i + 1, len(cluster_indices)):

                idx1 = cluster_indices[i]

                idx2 = cluster_indices[j]

                distance = distance_matrix[idx1, idx2]

                if distance < min_distance:

                    min_distance = distance

                    cluster1 = idx1

                    cluster2 = idx2

        if cluster2 in cluster_indices:

            cluster_indices.remove(cluster2)

        if cluster1 in cluster_indices:

            cluster_indices.remove(cluster1)

        cluster_indices.append(new_cluster_index)

        if new_cluster_index >= len(distance_matrix):

            new_size = new_cluster_index + 1

            new_matrix = np.zeros((new_size, new_size))

            new_matrix[:len(distance_matrix), :len(distance_matrix)] = distance_matrix

            distance_matrix = new_matrix

        Z[k, 0] = cluster1

        Z[k, 1] = cluster2

        Z[k, 2] = min_distance

        Z[k, 3] = len(cluster_indices)

        for i in range(len(cluster_indices)):

            if cluster_indices[i] != new_cluster_index:

                idx1 = cluster_indices[i]

                idx2 = new_cluster_index

                min_dist = min(distance_matrix[idx1, cluster1], distance_matrix[idx1, cluster2])

                distance_matrix[idx1, idx2] = min_dist

                distance_matrix[idx2, idx1] = min_dist

        new_cluster_index += 1



    return Z

# 0.4 Visualize HAC
def fig_hac(array, names):
    fig = plt.figure(figsize=(12, 6))
    dendrogram(array, labels=names, leaf_rotation=90)
    plt.tight_layout()
    
    return fig

# 0.5: Normalize features
def normalize_features(features):
    array = np.array(features)
    mean = np.mean(array, axis=0)
    sd = np.std(array, axis=0)
    norm_features = [(x - mean) / sd for x in features]
    
    return norm_features

# 0.6 Testing
if __name__ == "__main__":
    data = load_data("countries.csv")
    country_names = [row["Country"] for row in data]
    features = [calc_features(row) for row in data]
    features_normalized = normalize_features(features)
    n = 10
    Z_raw = hac(features[:n])
    Z_normalized = hac(features_normalized[:n])
    fig_raw = fig_hac(Z_raw, country_names[:n])
    plt.show()
