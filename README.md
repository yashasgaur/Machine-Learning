# Machine Learning — From Scratch & Scikit-learn
 
[![Python](https://img.shields.io/badge/python-%3E%3D3.8-blue)](https://www.python.org/)
 
A hands-on collection of fundamental machine learning algorithms implemented from first principles with NumPy and Python, alongside concise scikit-learn (and XGBoost/LightGBM) examples. This repository is designed to help you understand the math, intuition, and code behind core ML techniques while also showing how the industry-standard libraries implement them.
 
## Table of Contents
 
- [Overview](#overview)
- [Algorithms Included](#algorithms-included)
- [Repository Layout](#repository-layout)
- [Getting Started](#getting-started)
- [What You'll Learn](#what-youll-learn)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Author / Contact](#author--contact)
## Overview
 
This repo contains clear, well-documented implementations of classic machine learning algorithms. Each topic includes:
 
- A from-scratch implementation using NumPy and Python (where applicable)
- A scikit-learn (or XGBoost/LightGBM) comparison
- A runnable `.py` script and matching Jupyter `.ipynb` notebook
- The dataset used, included as a CSV in the same folder
The goal is not only to provide working code, but to make the underlying concepts intuitive and easy to study.
 
## Algorithms Included
 
The repository includes both foundational algorithm implementations and practical library-based references.
 
| # | Algorithm | From Scratch | Library |
| -- | ------------------------ | :----------: | :----------: |
| 01 | Linear Regression | `01-linear-regression-from-scratch/` | `02-linear-regression-sklearn/` |
| 02 | Logistic Regression | `03-logistic-regression-from-scratch/` | `04-logistic-regression-sklearn/` |
| 03 | K-Nearest Neighbors | `05-knn-from-scratch/` | `06-knn-sklearn/` |
| 04 | Decision Tree | `07-decision-tree-from-scratch/` | `08-decision-tree-sklearn/` |
| 05 | K-Means Clustering | `09-k-means-from-scratch/` | `10-k-means-sklearn/` |
| 06 | Random Forest | — | `11-random-forest-sklearn/` |
| 07 | Naive Bayes | — | `12-naive-bayes-sklearn/` |
| 08 | Support Vector Machine | — | `13-svm-sklearn/` |
| 09 | Principal Component Analysis | — | `14-pca-sklearn/` |
| 10 | Gradient Boosting (XGBoost & LightGBM) | — | `15-xgboost-lightgbm/` |
| 11 | DBSCAN (vs. K-Means) | — | `16-dbscan-sklearn/` |
 
### Linear Regression
 
Implemented from first principles using NumPy, compared against scikit-learn's `LinearRegression`.
 
Concepts covered: hypothesis function, mean squared error, cost function, gradient descent, learning rate, feature scaling, training over multiple epochs, cost convergence.
 
### Logistic Regression
 
Implemented as a binary classification model from scratch, validated against scikit-learn's `LogisticRegression`.
 
Concepts covered: linear decision function, sigmoid activation, log loss / binary cross-entropy, gradient descent, feature scaling, probability estimation, classification threshold, confusion matrix, precision/recall/F1-score.
 
### K-Nearest Neighbors
 
Implemented without a machine learning library for the core logic, compared with scikit-learn's `KNeighborsClassifier`.
 
Concepts covered: Euclidean distance, nearest-neighbor selection, choosing K, majority voting, feature scaling, classification.
 
### Decision Tree
 
A decision tree classifier implemented from scratch, compared against scikit-learn's `DecisionTreeClassifier`.
 
Concepts covered: recursive tree construction, feature splitting, Gini impurity, information gain / split quality, leaf nodes, prediction through tree traversal, stopping conditions.
 
### K-Means Clustering
 
Clustering implemented from scratch using NumPy, compared against scikit-learn's `KMeans`.
 
Concepts covered: cluster initialization, Euclidean distance, assignment step, centroid update step, iterative convergence, inertia / clustering objective.
 
### Random Forest (`11-random-forest-sklearn/`)
 
Trains a `RandomForestClassifier` on the Wine Quality dataset to predict wine quality from physicochemical features (acidity, sulphates, alcohol, etc.).
 
Concepts covered: ensemble learning, bagging, decision-tree aggregation, `n_estimators`.
 
### Naive Bayes (`12-naive-bayes-sklearn/`)
 
Trains `MultinomialNB` and `BernoulliNB` classifiers on an SMS spam dataset, using bag-of-words (`CountVectorizer`) features, and runs predictions on custom example messages.
 
Concepts covered: Bayes' theorem, conditional independence assumption, multinomial vs. Bernoulli variants, text vectorization, confusion matrix, classification report.
 
### Support Vector Machine (`13-svm-sklearn/`)
 
Trains `SVC` (linear and RBF kernels) on a social-network-ads dataset (age, estimated salary) to predict purchase behavior, and visualizes the decision boundary.
 
Concepts covered: maximum-margin classification, linear vs. RBF kernels, feature scaling, decision boundary visualization, accuracy scoring.
 
### Principal Component Analysis (`14-pca-sklearn/`)
 
Applies `PCA` to the Iris dataset to reduce four features down to two principal components for visualization.
 
Concepts covered: dimensionality reduction, feature scaling (`StandardScaler`), explained variance ratio, 2D projection and visualization.
 
### Gradient Boosting — XGBoost & LightGBM (`15-xgboost-lightgbm/`)
 
Trains `XGBClassifier` and `LGBMClassifier` side by side on the Wine Quality dataset and compares their accuracy and classification reports.
 
Concepts covered: gradient-boosted trees, `n_estimators`, `max_depth`, `learning_rate`, subsampling / column subsampling, model comparison.
 
### DBSCAN (`16-dbscan-sklearn/`)
 
Compares `KMeans` and `DBSCAN` on the same 2D dataset to show the difference between centroid-based and density-based clustering, including noise-point detection.
 
Concepts covered: density-based clustering, `eps` and `min_samples`, noise points / outliers, cluster visualization.
 
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
├── 13-svm-sklearn/
├── 14-pca-sklearn/
├── 15-xgboost-lightgbm/
├── 16-dbscan-sklearn/
│
└── README.md
```
 
Each folder contains a `.py` script, a matching `.ipynb` notebook, and the CSV dataset it uses.
 
## Getting Started
 
### 1. Clone the repository
 
```bash
git clone https://github.com/yashasgaur/Machine-Learning.git
cd Machine-Learning
```
 
### 2. Create and activate a virtual environment
 
```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows
```
 
### 3. Install dependencies
 
Each folder is self-contained; install the libraries it needs, for example:
 
```bash
pip install numpy pandas matplotlib scikit-learn xgboost lightgbm jupyter
```
 
### 4. Run a notebook
 
```bash
jupyter notebook
```
 
Then open the `.ipynb` file inside the algorithm folder you want to explore.
 
### 5. Or run a script directly
 
```bash
cd 01-linear-regression-from-scratch
python from_scratch.py
```
 
(Script filenames vary slightly by folder — e.g. `from_scratch.py`, `using_sklearn.py`, `using-sklearn.py`, `from_sklearn.py`, `boosting.py` — check the folder for the exact name.)
 
## What You'll Learn
 
This repository focuses on building a strong foundation in:
 
- Supervised learning (regression & classification)
- Unsupervised learning (clustering & dimensionality reduction)
- Gradient descent and optimization
- Ensemble methods and gradient boosting
- Feature scaling and preprocessing
- Model evaluation (accuracy, confusion matrix, precision/recall/F1)
- Algorithmic thinking and practical ML implementation
## Roadmap
 
Planned topics for this learning journey include:
 
- Neural networks (from scratch and with a deep learning framework)
- Model deployment / serving an ML model as an API
## Contributing
 
Contributions are welcome. If you want to:
 
- Open an issue for a bug or enhancement
- Suggest a new algorithm or improvement
- Submit a pull request with a short description
Please follow the standard GitHub contribution workflow.
 
## Author / Contact
 
**Author:** Yashas Gaur
 
**GitHub:** https://github.com/yashasgaur
 
