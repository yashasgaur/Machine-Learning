# Machine Learning — From Scratch & Scikit-learn

[![Python](https://img.shields.io/badge/python-%3E%3D3.8-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Requirements](https://img.shields.io/badge/requirements-.txt-lightgrey)](requirements.txt)

A hands-on collection of fundamental Machine Learning algorithms implemented from first principles with NumPy and Python, alongside concise Scikit-learn examples. This repository is designed to demonstrate your understanding of algorithm internals and to serve as a portfolio project for interviews and resumes.

## Table of Contents

- [About](#about)
- [Algorithms](#algorithms)
- [Repository Structure](#repository-structure)
- [Implementations](#implementations)
  - [From Scratch](#from-scratch)
  - [Scikit-learn](#scikit-learn)
- [Technologies](#technologies)
- [Usage](#usage)
- [Learning Focus](#learning-focus)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Author / Contact](#author--contact)

## About

This repo contains clear, well-documented implementations of classic ML algorithms. Each algorithm includes a "from-scratch" implementation (NumPy + Python) to explain the math and mechanics, plus a short Scikit-learn example to show the standard industry API and to validate results.

Each algorithm folder contains a short README, runnable notebooks or scripts, visualizations (where applicable), and a concise comparison section showing how the from-scratch output compares to Scikit-learn.

## Algorithms

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

Each algorithm implementation includes a concise Jupyter notebook or script demonstrating the algorithm, visualizations (where applicable), and a short comparison with Scikit-learn's implementation.

### From Scratch

The core mathematical and algorithmic logic is implemented using NumPy and Python. This is intended to help you understand how the algorithm works internally and how hyperparameters affect behavior.

### Scikit-learn

The corresponding Scikit-learn examples show the standard API usage and provide a practical comparison to the from-scratch code.

---

## Linear Regression

Implemented Linear Regression from first principles using NumPy.

Concepts covered:

- Hypothesis function
- Mean Squared Error
- Cost function
- Gradient Descent
- Learning Rate
- Feature Scaling
- Training over multiple epochs
- Cost convergence

The implementation is compared against Scikit-learn's `LinearRegression` implementation.

---

## Logistic Regression

Implemented binary Logistic Regression from scratch.

Concepts covered:

- Linear decision function
- Sigmoid activation
- Log Loss / Binary Cross-Entropy
- Gradient Descent
- Feature Scaling
- Probability estimation
- Classification threshold
- Confusion Matrix
- Precision, Recall and F1-score

The implementation is validated against Scikit-learn's `LogisticRegression`.

---

## K-Nearest Neighbors

Implemented KNN without using a machine-learning library for the core algorithm.

Concepts covered:

- Euclidean distance
- Nearest-neighbor selection
- Choosing K
- Majority voting
- Feature scaling
- Classification

The implementation is compared with Scikit-learn's `KNeighborsClassifier`.

---

## Decision Tree

Implemented a Decision Tree classifier from scratch.

Concepts covered:

- Recursive tree construction
- Feature splitting
- Gini impurity
- Information gain / split quality
- Leaf nodes
- Prediction through tree traversal
- Stopping conditions

The implementation is compared against Scikit-learn's `DecisionTreeClassifier`.

---

## K-Means Clustering

Implemented K-Means clustering from scratch using NumPy.

Concepts covered:

- Cluster initialization
- Euclidean distance
- Assignment step
- Centroid update step
- Iterative convergence
- Inertia / clustering objective

The implementation is compared against Scikit-learn's `KMeans`.

## Technologies

- Python 3.8+
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook

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

3. Run notebooks:

```bash
jupyter notebook
# then open the notebook files under each algorithm folder
```

4. Or run example scripts (where provided):

```bash
python 01-linear-regression-from-scratch/example.py
```

## Learning Focus

This repository focuses on building strong foundations in:

- Supervised Learning
- Regression
- Classification
- Optimization
- Gradient Descent
- Feature Scaling
- Model Evaluation
- Algorithmic Thinking
- Understanding ML implementations

## Roadmap

Planned topics for this learning journey include:

- Random Forest
- Naive Bayes
- Support Vector Machines
- Principal Component Analysis
- Gradient Boosting

---

## Contributing

Contributions are welcome. If you want to:

- Open an issue for a bug or enhancement
- Suggest new algorithms or improvements
- Send a pull request with tests and a short description

Please follow standard GitHub contribution workflows.

## License

This repository is released under the MIT License. See the [LICENSE](LICENSE) file for details.

## Author / Contact

**Author:** Yashas Gaur
**GitHub:** https://github.com/yashasgaur

If you'd like a specific resume blurb, LinkedIn, or email added to the README, tell me and I'll include it.