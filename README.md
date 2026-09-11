# Machine Learning — From Scratch & Scikit-learn

[![Python](https://img.shields.io/badge/python-%3E%3D3.8-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A hands-on implementation of fundamental Machine Learning algorithms, built from first principles using Python and NumPy and then compared with their Scikit-learn equivalents.

The goal of this repository is to understand **how core ML algorithms work internally**, rather than treating them as black-box library functions.

## Table of Contents

- [Algorithms](#algorithms)
- [Repository Structure](#repository-structure)
- [Implementations](#implementations)
  - [From Scratch](#from-scratch)
  - [Scikit-learn](#scikit-learn)
- [Technologies](#technologies)
- [Usage](#usage)
- [Learning Focus](#learning-focus)
- [Future Learning](#future-learning)
- [Author](#author)

## Algorithms

This repo contains the following algorithms (each implemented from-scratch and with a Scikit-learn example where applicable):

| #  | Algorithm                | From Scratch | Scikit-learn |
| -- | ------------------------ | :----------: | :----------: |
| 01 | Linear Regression        | 01-linear-regression-from-scratch/ | 02-linear-regression-sklearn/ |
| 02 | Logistic Regression      | 03-logistic-regression-from-scratch/ | 04-logistic-regression-sklearn/ |
| 03 | K-Nearest Neighbors      | 05-knn-from-scratch/ | 06-knn-sklearn/ |
| 04 | Decision Tree            | 07-decision-tree-from-scratch/ | 08-decision-tree-sklearn/ |
| 05 | K-Means Clustering       | 09-k-means-from-scratch/ | 10-k-means-sklearn/ |

## Repository Structure

```text
Machine-Learning/
│
├── 01-linear-regression-from-scratch/
├── 02-linear-regression-sklearn/
│
├── 03-logistic-regression-from-scratch/
├── 04-logistic-regression-sklearn/
│
├── 05-knn-from-scratch/
├── 06-knn-sklearn/
│
├── 07-decision-tree-from-scratch/
├── 08-decision-tree-sklearn/
│
├── 09-k-means-from-scratch/
└── 10-k-means-sklearn/
```

## Implementations

Each algorithm implementation includes a `notebook` or `script` demonstrating the algorithm, visualizations (where applicable), and a comparison with Scikit-learn's implementation.

### From Scratch

The core mathematical and algorithmic logic is implemented manually using NumPy and Python. This is intended to develop an understanding of how the algorithm works internally, how parameters are updated during training, and how predictions are generated.

### Scikit-learn

The corresponding Scikit-learn implementation shows the standard API usage and provides a practical comparison to the from-scratch code.

---

### Linear Regression

Implemented Linear Regression from first principles using NumPy.

Concepts covered:

* Hypothesis function
* Mean Squared Error
* Cost function
* Gradient Descent
* Learning Rate
* Feature Scaling
* Training over multiple epochs
* Cost convergence

The implementation is compared against Scikit-learn's `LinearRegression` implementation.

---

### Logistic Regression

Implemented binary Logistic Regression from scratch.

Concepts covered:

* Linear decision function
* Sigmoid activation
* Log Loss / Binary Cross-Entropy
* Gradient Descent
* Feature Scaling
* Probability estimation
* Classification threshold
* Confusion Matrix
* Precision, Recall and F1-score

The implementation is validated against Scikit-learn's `LogisticRegression`.

---

### K-Nearest Neighbors

Implemented KNN without using a machine-learning library for the core algorithm.

Concepts covered:

* Euclidean distance
* Nearest-neighbor selection
* Choosing K
* Majority voting
* Feature scaling
* Classification

The implementation is compared with Scikit-learn's `KNeighborsClassifier`.

---

### Decision Tree

Implemented a Decision Tree classifier from scratch.

Concepts covered:

* Recursive tree construction
* Feature splitting
* Gini impurity
* Information gain / split quality
* Leaf nodes
* Prediction through tree traversal
* Stopping conditions

The implementation is compared against Scikit-learn's `DecisionTreeClassifier`.

---

### K-Means Clustering

Implemented K-Means clustering from scratch using NumPy.

Concepts covered:

* Cluster initialization
* Euclidean distance
* Assignment step
* Centroid update step
* Iterative convergence
* Inertia / clustering objective

The implementation is compared against Scikit-learn's `KMeans`.

## Technologies

* Python
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* Jupyter Notebook

## Usage

1. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Open the notebooks in Jupyter or run the example scripts in each folder.

## Learning Focus

This repository focuses on building strong foundations in:

* Supervised Learning
* Regression
* Classification
* Optimization
* Gradient Descent
* Feature Scaling
* Model Evaluation
* Algorithmic Thinking
* Understanding ML implementations

## Why From Scratch?

Machine-learning libraries make model training extremely convenient, but they can hide the underlying mechanics. Implementing these algorithms from scratch helps understand what happens behind APIs such as:

```python
model.fit(X_train, y_train)
```

The Scikit-learn implementations are then used to connect that theoretical understanding with standard industry tooling.

## Future Learning

Planned topics for this learning journey include:

* Random Forest
* Naive Bayes
* Support Vector Machines
* Principal Component Analysis
* Gradient Boosting

---

**Author:** Yashas Gaur  
**GitHub:** https://github.com/yashasgaur
