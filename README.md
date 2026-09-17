# Machine Learning — From Scratch & Scikit-learn

[![Python](https://img.shields.io/badge/python-%3E%3D3.8-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Requirements](https://img.shields.io/badge/requirements-.txt-lightgrey)](requirements.txt)

A hands-on collection of fundamental machine learning algorithms implemented from first principles with NumPy and Python, alongside concise scikit-learn examples. This repository is designed to help you understand the math, intuition, and code behind core ML techniques while also showing how the industry-standard library implements them.

## Table of Contents

- [Overview](#overview)
- [Algorithms Included](#algorithms-included)
- [Repository Layout](#repository-layout)
- [Getting Started](#getting-started)
- [What You'll Learn](#what-youll-learn)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Author / Contact](#author--contact)

## Overview

This repo contains clear, well-documented implementations of classic machine learning algorithms. Each topic includes:

- A from-scratch implementation using NumPy and Python
- A scikit-learn comparison where relevant
- Short explanations of the underlying mathematics
- Practical examples and visualizations where useful
- Guidance on model behavior, assumptions, and evaluation

The goal is not only to provide working code, but to make the underlying concepts intuitive and easy to study.

## Algorithms Included

The repository includes both foundational algorithm implementations and practical scikit-learn references.

| # | Algorithm | From Scratch | scikit-learn |
| -- | ------------------------ | :----------: | :----------: |
| 01 | Linear Regression | 01-linear-regression-from-scratch/ | 02-linear-regression-sklearn/ |
| 02 | Logistic Regression | 03-logistic-regression-from-scratch/ | 04-logistic-regression-sklearn/ |
| 03 | K-Nearest Neighbors | 05-knn-from-scratch/ | 06-knn-sklearn/ |
| 04 | Decision Tree | 07-decision-tree-from-scratch/ | 08-decision-tree-sklearn/ |
| 05 | K-Means Clustering | 09-k-means-from-scratch/ | 10-k-means-sklearn/ |
| 11 | Random Forest | - | 11-random-forest-sklearn/ |
| 12 | Naive Bayes | - | 12-naive-bayes-sklearn/ |

### Linear Regression

Implemented from first principles using NumPy.

Concepts covered:

- Hypothesis function
- Mean squared error
- Cost function
- Gradient descent
- Learning rate
- Feature scaling
- Training over multiple epochs
- Cost convergence

This implementation is compared against scikit-learn's `LinearRegression`.

### Logistic Regression

Implemented as a binary classification model from scratch.

Concepts covered:

- Linear decision function
- Sigmoid activation
- Log loss / binary cross-entropy
- Gradient descent
- Feature scaling
- Probability estimation
- Classification threshold
- Confusion matrix
- Precision, recall, and F1-score

This implementation is validated against scikit-learn's `LogisticRegression`.

### K-Nearest Neighbors

Implemented without using a machine learning library for the core logic.

Concepts covered:

- Euclidean distance
- Nearest-neighbor selection
- Choosing K
- Majority voting
- Feature scaling
- Classification

This is compared with scikit-learn's `KNeighborsClassifier`.

### Decision Tree

Implemented a decision tree classifier from scratch.

Concepts covered:

- Recursive tree construction
- Feature splitting
- Gini impurity
- Information gain / split quality
- Leaf nodes
- Prediction through tree traversal
- Stopping conditions

This is compared against scikit-learn's `DecisionTreeClassifier`.

### K-Means Clustering

Implemented clustering from scratch using NumPy.

Concepts covered:

- Cluster initialization
- Euclidean distance
- Assignment step
- Centroid update step
- Iterative convergence
- Inertia / clustering objective

This is compared against scikit-learn's `KMeans`.

### Random Forest (scikit-learn example)

A scikit-learn example implementing random forest classification and regression with `RandomForestClassifier` and `RandomForestRegressor`.

Concepts covered:

- Ensemble learning
- Bagging
- Feature importance
- Hyperparameter tuning (`n_estimators`, `max_depth`)
- Out-of-bag evaluation (when applicable)

See `11-random-forest-sklearn/` for notebooks and example scripts.

### Naive Bayes (scikit-learn example)

A scikit-learn example demonstrating Naive Bayes classifiers, including Gaussian and multinomial variants where applicable.

Concepts covered:

- Bayes' theorem
- Conditional independence assumption
- Gaussian vs. multinomial vs. Bernoulli variants
- Probabilistic predictions
- Use cases and limitations

See `12-naive-bayes-sklearn/` for notebooks and example scripts.

## Repository Layout

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
├── 10-k-means-sklearn/
│
├── 11-random-forest-sklearn/
├── 12-naive-bayes-sklearn/
├── LICENSE
├── README.md
├── requirements.txt
└── .gitignore
```

## Getting Started

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run notebooks

```bash
jupyter notebook
```

Then open the notebook files in each algorithm folder.

### 4. Run example scripts

```bash
python 01-linear-regression-from-scratch/example.py
```

## What You'll Learn

This repository focuses on building a strong foundation in:

- Supervised learning
- Regression
- Classification
- Optimization
- Gradient descent
- Feature scaling
- Model evaluation
- Algorithmic thinking
- Practical ML implementation

## Roadmap

Planned topics for this learning journey include:

- Support Vector Machines
- Principal Component Analysis
- Gradient Boosting
- Neural Networks

## Contributing

Contributions are welcome. If you want to:

- Open an issue for a bug or enhancement
- Suggest a new algorithm or improvement
- Submit a pull request with tests and a short description

Please follow the standard GitHub contribution workflow.

## License

This repository is released under the MIT License. See the [LICENSE](LICENSE) file for details.

## Author / Contact

**Author:** Yashas Gaur

**GitHub:** https://github.com/yashasgaur

If you'd like a resume blurb, LinkedIn link, or email added to the README, let me know and I can include it.
