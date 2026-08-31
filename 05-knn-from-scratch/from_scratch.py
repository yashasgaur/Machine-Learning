import numpy as np
import pandas as pd
from collections import Counter

columns = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "species"
]

data = pd.read_csv("iris.csv", names = columns)
indices = np.random.permutation(len(data))
data = data.iloc[indices]

train_data = data.iloc[:120]
test_data = data.iloc[120:]

X_train = train_data[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
y_train = train_data["species"]

X_test = test_data[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
y_test = test_data["species"]

# Standardizing X_train and X_test by X_train's stats

column_sum = np.sum(X_train, axis = 0)
mean = column_sum / len(X_train)

sum2 = X_train - mean
sum2sq = np.sum(sum2 ** 2, axis = 0)
variance = sum2sq / len(X_train)

sd = variance ** (0.5)

X_train = sum2 / sd
sum_test = X_test - mean
X_test = sum_test / sd

# X_train and X_test now standardized

def distance(training_flower, test_flower):
    difference = training_flower - test_flower
    square = difference ** 2
    squared_sum = np.sum(square)
    root = squared_sum ** 0.5
    
    return root
    

distance(X_train.iloc[0], X_test.iloc[0])

def predict_one(test_flower, k=3):
    distances = []

    for i in range(len(X_train)):
        training_flower = X_train.iloc[i]
        distances.append(distance(training_flower, test_flower))
        
    neighbours = list(zip(distances, y_train))
    neighbours.sort()
    
    nearest = neighbours[:k]

    neighbor_species = []
    for i in range(len(nearest)):
        neighbor_species.append(nearest[i][1])
    
    counts = Counter(neighbor_species)
    prediction = counts.most_common(1)
    
    return prediction[0][0]

predictions = []

for i in range(len(X_test)):
    prediction = predict_one(X_test.iloc[i])

    predictions.append(prediction)
    
print(predictions)

# Accuracy

correct = 0
for i in range(len(predictions)):
    if predictions[i] == y_test.iloc[i]:
        correct += 1
        
accuracy = (correct / len(predictions)) * 100
print(str(accuracy) + "%")

# Predict a new flower

sepal_length = float(input("Enter sepal length: "))
sepal_width = float(input("Enter sepal width: "))
petal_length = float(input("Enter petal length: "))
petal_width = float(input("Enter petal width: "))

new_flower = np.array([
    sepal_length,
    sepal_width,
    petal_length,
    petal_width
])

# Standardize using X_train's mean and SD
new_flower = (new_flower - mean) / sd

prediction = predict_one(new_flower)

print("Predicted species:", prediction)

