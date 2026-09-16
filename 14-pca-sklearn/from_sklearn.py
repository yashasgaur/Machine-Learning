import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

columns = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "species"
]

data = pd.read_csv("iris.csv", names = columns)

X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = data['species']

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

print(pca.explained_variance_ratio_)

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=pd.factorize(y)[0]
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA - Iris Dataset")

plt.show()

