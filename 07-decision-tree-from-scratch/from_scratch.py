import numpy as np
import pandas as pd
from collections import Counter

data = pd.read_csv("BankNote_Authentication.csv")
data.info()

indices = np.random.permutation(len(data)) # gets indices col in random order
data = data.iloc[indices]

train_split = int(len(data) * 0.8)
train_data = data.iloc[:train_split]
test_data = data.iloc[train_split:]

X_train = train_data[["variance", "skewness", "curtosis", "entropy"]]
y_train = train_data["class"]

X_test = test_data[["variance", "skewness", "curtosis", "entropy"]]
y_test = test_data["class"]

def gini(labels):
    count = Counter(labels)
    summition = 0
    
    def probability(count):
        prob = count / len(labels)
        return prob

    def squares(probability):
        return probability ** 2
    
    for value in count.values():
        summition += squares(probability(value))
        
    return 1 - summition

def find_best_split(X_train, y_train):

    best_gini = float("inf")
    best_threshold = None
    best_feature = None

    for feature in X_train.columns:
        for threshold in X_train[feature].unique():

            left_mask = X_train[feature] <= threshold
            right_mask = X_train[feature] > threshold
            
            left_labels = y_train[left_mask]
            right_labels = y_train[right_mask]

            if len(left_labels) == 0 or len(right_labels) == 0:
                continue
            
            n = len(left_labels) + len(right_labels)

            weighted_gini = (
                (len(left_labels) / n) * gini(left_labels)
                +
                (len(right_labels) / n) * gini(right_labels)
            )

            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_threshold = threshold
                best_feature = feature

    return best_threshold, best_gini, best_feature

def build_tree(X, y, depth=0, max_depth=5):

    if len(Counter(y)) == 1 or depth == max_depth:
        return Counter(y).most_common(1)[0][0]

    best_threshold, best_gini, best_feature = find_best_split(X, y)

    left_mask = X[best_feature] <= best_threshold
    right_mask = X[best_feature] > best_threshold

    X_left = X[left_mask]
    X_right = X[right_mask]

    y_left = y[left_mask]
    y_right = y[right_mask]

    left_tree = build_tree(X_left, y_left, depth + 1, max_depth)
    right_tree = build_tree(X_right, y_right, depth + 1, max_depth)

    return {
        "feature": best_feature,
        "threshold": best_threshold,
        "left": left_tree,
        "right": right_tree
    }

tree = build_tree(X_train, y_train)

def predict_one(row, tree):

    if not isinstance(tree, dict):
        return tree

    feature = tree["feature"]
    threshold = tree["threshold"]

    if row[feature] <= threshold:
        return predict_one(row, tree["left"])
    else:
        return predict_one(row, tree["right"])

predictions = []

for i in range(len(X_test)):
    prediction = predict_one(X_test.iloc[i], tree)
    predictions.append(prediction)
    
accuracy = np.mean(np.array(predictions) == np.array(y_test))

print("Accuracy:", accuracy)