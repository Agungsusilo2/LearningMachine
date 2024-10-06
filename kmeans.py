import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


data = pd.read_csv('data.csv')

columns = ["Umur", "Penghasilan", "Jangka Perpindahan", "Kualitas Barang", "Kesadaran Limbah"]


def calculate_euclidean_distance(df, centroid, columns):
    distances = []
    for i in range(len(df)):
        row = df.loc[i]
        distance = np.sqrt(sum((row[col] - centroid[col]) ** 2 for col in columns))
        distances.append(distance)
    return distances


def update_centroids(df, columns):
    return df.groupby('Cluster')[columns].mean().reset_index()


initial_centroids_indices = [4, 11, 18]
centroids = data.loc[initial_centroids_indices, columns].reset_index(drop=True)

iterations = 3

for iteration in range(iterations):
    distances_to_c1 = calculate_euclidean_distance(data, centroids.loc[0], columns)
    distances_to_c2 = calculate_euclidean_distance(data, centroids.loc[1], columns)
    distances_to_c3 = calculate_euclidean_distance(data, centroids.loc[2], columns)

    df_distances = pd.DataFrame({
        "Umur": data[columns[0]],
        "Penghasilan": data[columns[1]],
        "Jangka Perpindahan": data[columns[2]],
        "Kualitas Barang": data[columns[3]],
        "Kesadaran Limbah": data[columns[4]],
        'Distances_to_c1': distances_to_c1,
        'Distances_to_c2': distances_to_c2,
        'Distances_to_c3': distances_to_c3
    })

    df_distances['Cluster'] = df_distances[['Distances_to_c1', 'Distances_to_c2', 'Distances_to_c3']].idxmin(axis=1)
    df_distances['Cluster'] = df_distances['Cluster'].apply(lambda x: int(x.split('_')[-1][-1]))

    centroids = update_centroids(df_distances, columns)
    # print(f"Iteration {iteration + 1} centroids:")
    # print(centroids)

df_distances.to_csv('distances_to_centroids.csv', index=False)

cluster_means = update_centroids(df_distances, columns)
df_distances[df_distances['Cluster'] == 1].to_csv('cluster_means1.csv', index=False)
df_distances[df_distances['Cluster'] == 2].to_csv('cluster_means2.csv', index=False)
df_distances[df_distances['Cluster'] == 3].to_csv('cluster_means3.csv', index=False)


total_cluster_1 = df_distances[df_distances['Cluster'] == 1].shape[0]
total_cluster_2 = df_distances[df_distances['Cluster'] == 2].shape[0]
total_cluster_3 = df_distances[df_distances['Cluster'] == 3].shape[0]
print(total_cluster_1, total_cluster_2, total_cluster_3)

plt.figure(figsize=(10, 7))

sns.scatterplot(data=df_distances, x="Umur", y="Penghasilan", hue="Cluster", s=100)

# Plot centroids
for i, centroid in centroids.iterrows():
    plt.scatter(centroid["Umur"], centroid["Penghasilan"], s=200, c='black', marker='X')

for i, row in df_distances.iterrows():
    centroid = centroids.loc[row['Cluster'] - 1]
    plt.plot([row["Umur"], centroid["Umur"]], [row["Penghasilan"], centroid["Penghasilan"]], c='grey', linestyle='--')

plt.title('Clusters of Data Points with Centroids and Distance Lines')
plt.xlabel('Umur')
plt.ylabel('Penghasilan')

plt.savefig('clusters_with_centroids_and_distance_lines.png')
plt.show()