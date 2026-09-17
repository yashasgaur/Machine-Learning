import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN

data = pd.read_csv("dbscan_dataset.csv")

X = data[["Feature_1", "Feature_2"]]

kmeans = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
)

data["KMeans_Cluster"] = kmeans.fit_predict(X)

dbscan = DBSCAN(
    eps=0.2,
    min_samples=5
)

data["DBSCAN_Cluster"] = dbscan.fit_predict(X)

plt.scatter(
    X["Feature_1"],
    X["Feature_2"],
    c=data["KMeans_Cluster"]
)

plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

plt.scatter(
    X["Feature_1"],
    X["Feature_2"],
    c=data["DBSCAN_Cluster"]
)

plt.title("DBSCAN Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

noise_points = np.sum(data["DBSCAN_Cluster"] == -1)

print("Noise points:", noise_points)



