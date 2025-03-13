# Machine Learning Model Evaluation

## Overview
This project implements and evaluates two fundamental machine learning models—Logistic Regression and Naive Bayes Gaussian—to understand their performance in binary classification tasks.
Through feature selection, cross-validation, and decision boundary visualization, this project provides insights into how these models learn from data, optimize parameters, and generalize to unseen examples.
The goal is to compare the strengths and limitations of each approach and explore how they perform under different dataset conditions.

## Key Features
Implementation from Scratch – Logistic Regression with Gradient Descent and Naive Bayes using Gaussian Mixture Models (EM Algorithm).
Feature Selection – Using Pearson correlation to identify the most relevant features.
Cross-Validation – Assessing model performance with multiple train-test splits.
Performance Metrics – Training and test accuracy comparison for both models.
Visualizations –
  Decision Boundaries: Observe how models classify different regions.
  Convergence Graphs: Track cost function reduction over iterations for Logistic Regression.

## Data Preparation
The dataset is not included in this repository. You must provide your own training and test datasets in CSV format.

## Results & Insights
Logistic Regression:
  Minimizes the cost function over iterations using gradient descent.
  More robust with well-separated data but can struggle with non-linearly separable cases.
Naive Bayes (GMM with EM Algorithm)
  Approximates distributions using Gaussian Mixture Models (GMM).
  Performs well with overlapping data distributions and provides probabilistic predictions.
