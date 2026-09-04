import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = pd.read_csv("Mall_Customers.csv")
data = data.drop(columns=["CustomerID", "Gender", "Age"])

X = data.values

k = 6
model = KMeans(n_clusters=k)
model.fit(X)
model.labels_
for i in range(k):
    points = X[model.labels_ == i]
    plt.scatter(points[:, 0], points[:, 1])
    
plt.scatter(
    model.cluster_centers_[:, 0],
    model.cluster_centers_[:, 1],
    marker="X"
)

wcss = []

for i in range(1, 11):
    model = KMeans(n_clusters=i)
    model.fit(X)
    wcss.append(model.inertia_)

plt.plot(range(1, 11), wcss, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()



