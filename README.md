# Term Deposit Marketing Campaign Analysis

## Overview

This project involves analyzing a dataset from a European banking institution's direct marketing campaign. The goal is to predict whether a customer will subscribe to a term deposit based on various features. The analysis employs a two-layer machine learning approach to improve the success rate of marketing calls and provides insights into customer segments that are more likely to subscribe.

## Business Problem

A fintech company aims to enhance the success rate of calls made to customers for term deposit subscriptions. The project focuses on designing a machine learning model that not only predicts customer subscriptions with high accuracy but also offers interpretability to help clients make informed decisions.

## Badges

![Python version](https://img.shields.io/badge/python-3.x-blue.svg)
![repo-size](https://img.shields.io/github/languages/code-size/bellapp/fLJZ837zANOP6KDD?color=green)


## Data Description

The dataset includes information from direct marketing efforts, with features such as:
- **Age**: Customer's age
- **Job**: Type of job
- **Marital**: Marital status
- **Education**: Education level
- **Default**: Has credit in default?
- **Balance**: Average yearly balance in euros
- **Housing**: Has a housing loan?
- **Loan**: Has a personal loan?
- **Contact**: Contact communication type
- **Day**: Last contact day of the month
- **Month**: Last contact month of the year
- **Duration**: Last contact duration in seconds
- **Campaign**: Number of contacts performed during this campaign
- **Target (y)**: Has the client subscribed to a term deposit?

## Main Goals

- Predict if a customer will subscribe to a term deposit.
- Achieve an accuracy of 81% or above using 5-fold cross-validation.
- Identify customer segments more likely to subscribe.
- Determine key features influencing customer decisions.

## Approach

### Layer 1: Model for Selecting Customers to Call
- **Objective**: Reduce the number of calls by identifying customers likely to subscribe, the main score metric to compare best models is recall(1).
- **Best Model**: ExtraTreesClassifier with a recall(1) score of 0.73 for the test set and 0.87 for the training set on a balanced dataset.

### Layer 2: Model for Predicting Term Deposit Subscription
- **Objective**: Predict subscription using the entire dataset,the main score metric to compare best models is precision(1).
- **Best Model**: ExtraTreesClassifier with a precision(1) score of 0.81.

## Methodology

1. **Data Preprocessing**: Handling missing values, encoding categorical variables, and scaling numerical features.
2. **Exploratory Data Analysis (EDA)**: Understanding data distribution and relationships between features.
3. **Model Training**: Using LazyPredict, PyCaret, TPOT, and Optuna for model selection and hyperparameter tuning.
4. **Clustering**: Identifying customer segments using KMeans and PCA for visualization.

## Results

- The ExtraTreesClassifier was identified as the best model for both layers, providing a balance between precision and recall.
- The model successfully reduced the number of calls needed by identifying key customer segments.
- The solution will bring many benefits:
    - The total number of people to contact will be 4775 instead of 40000.
    - We will save more than 2200 hours in call time.
  




## Clustering Analysis

This section of the project focuses on clustering analysis to identify distinct groups within the customer data who have subscribed to term deposits. The analysis employs several techniques to determine the optimal number of clusters and visualize the results.

### Data Preprocessing

- **Numerical Features**: Standardized using `StandardScaler`.
- **Categorical Features**: Encoded using `OneHotEncoder` with the first category dropped to avoid multicollinearity.

### K-Means Clustering

1. **Elbow Method**: Used to determine the optimal number of clusters by plotting the inertia for different values of k.
2. **Silhouette Analysis**: Evaluated the silhouette score for different numbers of clusters to assess the quality of clustering.
3. **Optimal Clusters**: Based on the analysis, the optimal number of clusters was determined to be 3.
4. **Cluster Characteristics**: 
   - Cluster 0: Predominantly management jobs, married, secondary education.
   - Cluster 1: Predominantly management jobs, married, tertiary education.
   - Cluster 2: Predominantly blue-collar jobs, married, secondary education.

### Dimensionality Reduction and Visualization

- **PCA (Principal Component Analysis)**: 
  - Reduced data to 2 and 3 dimensions for visualization.
  - Visualized clusters in 2D and 3D space to understand the distribution and separation of clusters.

- **t-SNE (t-Distributed Stochastic Neighbor Embedding)**:
  - Applied for 2D and 3D visualization to capture non-linear relationships in the data.

- **UMAP (Uniform Manifold Approximation and Projection)**:
  - Used for 2D and 3D visualization to maintain the global structure of the data.

### Conclusion

The first part of the project demonstrates the effectiveness of a two-layer machine learning approach in improving marketing campaign outcomes. By focusing on key features and customer segments, the model provides actionable insights for targeted marketing strategies.

The clustering analysis part revealed distinct customer segments based on their demographic and financial attributes. These insights can be used to tailor marketing strategies and improve the targeting of potential customers for term deposit subscriptions. The use of multiple visualization techniques ensures a comprehensive understanding of the data structure and cluster characteristics.


## Requirements

- Python 3.x
- Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, pycaret, plotly, imbalanced-learn, hyperopt, optuna, tpot

## How to Use

1. Clone the repository.
2. Install the required libraries using `pip install -r requirements.txt`.
3. Run the Jupyter notebook `1_Term_deposit_Marketing_Modeling.ipynb` to reproduce the analysis.
4. Run the Jupyter notebook `2_Term_deposit_Marketing_Clustering.ipynb` to reproduce the clustering.