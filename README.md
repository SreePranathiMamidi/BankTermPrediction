# Bank Term Deposit Prediction Project

## Overview
This project leverages **Machine Learning** and **Deep Learning** techniques to predict whether a customer will subscribe to a bank term deposit based on their demographic and transactional data. The implementation involves feature engineering, data preprocessing, and model training using **Scikit-learn** and **TensorFlow/Keras**.

---

## Table of Contents
- [Overview](#overview)
- [Technologies Used](#technologies-used)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Workflow](#project-workflow)
- [Results](#results)
- [References](#references)

---

## Technologies Used
### Python Libraries
- **Scikit-learn**
  - For preprocessing, feature selection, and machine learning models.
- **TensorFlow/Keras**
  - For constructing and training neural networks.
- **Seaborn & Matplotlib**
  - For data visualization.
- **Pandas & NumPy**
  - For data manipulation and numerical computations.

---

## Features
### Key Functionalities
1. **Data Preprocessing**:
   - Encoding categorical features using `OrdinalEncoder`.
   - Normalizing numerical features with `MinMaxScaler`.
2. **Feature Engineering**:
   - Selecting top features using `SelectKBest` with chi-squared scoring.
3. **Machine Learning**:
   - Implementing a multi-layer perceptron (MLP) classifier using Scikit-learn.
4. **Deep Learning**:
   - Designing and training a neural network using TensorFlow/Keras.
   - Visualizing the model architecture and training performance.
5. **Evaluation**:
   - Assessing models with metrics like accuracy, confusion matrix, and learning curves.

---

## Dataset
The dataset used is **bank-full.csv**, containing customer data and target labels indicating subscription to a bank term deposit. It includes:
- **Categorical Features**: `job`, `marital`, `education`, etc.
- **Numerical Features**: `age`, `balance`, `duration`, etc.
- **Target Variable**: Binary (`yes`/`no`) indicating subscription.

---

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.7+
- Jupyter Notebook or Google Colab

### Steps
1. Clone the repository or download the files.
   ```bash
   git clone https://github.com/your-repo/bank-term-prediction.git
   ```
2. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Place the dataset (`bank-full.csv`) in the project directory.

---

## Project Workflow
1. **Data Loading**
   - Load the `bank-full.csv` dataset.
2. **Exploratory Data Analysis (EDA)**
   - Visualize feature distributions and correlations.
3. **Feature Engineering**
   - Encode categorical features and normalize numerical data.
   - Select top features based on importance scores.
4. **Model Building**
   - Train an MLPClassifier (Scikit-learn).
   - Develop a deep learning model with Dense and Dropout layers (TensorFlow).
5. **Evaluation and Visualization**
   - Plot learning curves and evaluate model accuracy.

---

## Results
- **Best Features**: Identified using chi-squared scoring.
- **Model Accuracy**:
  - **Scikit-learn MLPClassifier**: Achieved an accuracy of ~XX%.
  - **TensorFlow Neural Network**: Achieved an accuracy of ~XX%.
- **Learning Curve**: Demonstrates convergence of training and validation loss.

---

## References
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Original Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
