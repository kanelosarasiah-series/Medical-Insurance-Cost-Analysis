# Medical Insurance Cost Analysis and Prediction

## Project Overview

This project analyses medical insurance costs using data analytics, exploratory data analysis (EDA), and machine learning techniques.

The main aim of the project is to understand the factors associated with medical insurance charges and to build machine learning models that can predict insurance costs.

## Dataset

The dataset contains information about medical insurance customers.

The main variables include:

- Age
- Sex
- BMI
- Number of children
- Smoking status
- Region
- Medical insurance charges

The target variable for the prediction models is `charges`.

## Project Objectives

The objectives of this project are to:

1. Explore and understand the medical insurance dataset.
2. Clean and prepare the data for analysis.
3. Identify patterns and relationships between customer characteristics and insurance charges.
4. Create visualisations to communicate important findings.
5. Build and compare machine learning regression models.
6. Evaluate model performance using MAE, RMSE and R².
7. Use clustering to identify broad customer segments.
8. Examine feature importance to understand which variables contribute most to predictions.

## Technologies and Tools

The project was developed using:

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook
- Visual Studio Code

## Data Analysis

The project includes exploratory data analysis to examine:

- The distribution of insurance charges
- Age and insurance charges
- BMI and insurance charges
- Smoking status and insurance charges
- Correlations between numerical variables
- Other patterns within the dataset

The analysis showed that smoking status was strongly associated with differences in average insurance charges. Age and BMI also showed positive relationships with charges, while the number of children showed a much weaker relationship.

These findings describe patterns in the dataset and should not be interpreted as proof of causation.

## Machine Learning Models

Three regression models were developed and compared:

1. Linear Regression
2. Ridge Regression
3. Random Forest Regression

The models were evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

## Model Results

The test-set results were:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 1860.92 | 2436.64 | 0.965 |
| Ridge Regression | 1861.63 | 2438.83 | 0.965 |
| Random Forest Regression | 1297.03 | 1642.72 | 0.984 |

Five-fold cross-validation was also performed, with a mean R² of approximately 0.982.

## Clustering

K-Means clustering was used to identify broad customer segments based on selected customer characteristics.

Four clusters were explored to understand differences between groups of customers.

## Feature Importance

Feature importance analysis was used to investigate which variables contributed most to the model's predictions.

This provides additional insight into the factors associated with predicted insurance charges.

## Project Files

The repository contains:

```text
Medical_Insurance
│
├── Data
├── Results
├── Presentation
├── insurance_medical_cost.ipynb
└── README.md