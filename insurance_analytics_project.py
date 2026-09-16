# %%


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("Libraries imported successfully!")



# %%
# Load the dataset
df = pd.read_csv("Data/insurance_medical_cost.csv")

print("Dataset loaded successfully!")



# %%
# Display the first 5 rows
print("\nFirst 5 rows of the dataset:")
print(df.head())

# %%
print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

# %%
# Number of rows and columns
print("\nNumber of rows and columns:")
print(df.shape)


# %%
# Column names
print("\nColumn names:")
print(df.columns.tolist())

# %%
# Data types
print("\nData types:")
print(df.dtypes)

# %%
# Dataset information
print("\nDetailed information:")
df.info()

# %%
# Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# %%
# Check duplicate rows
print("\nDuplicate rows:")
print(df.duplicated().sum())

# %%
# Clean The Data
print("=" * 60)
print("DATA CLEANING")
print("=" * 60)

# %%
# Check for missing values
missing_values = df.isnull().sum()

print("\nMissing values before cleaning:")
print(missing_values)

# %%
# Remove duplicate rows
duplicates = df.duplicated().sum()

if duplicates > 0:
    df = df.drop_duplicates()
    print(f"\nRemoved {duplicates} duplicate rows.")
else:
    print("\nNo duplicate rows found.")


# %%
# Check again
print("\nDataset shape after cleaning:")
print(df.shape)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nCleaning completed successfully!")

# %%
# Descriptive Statistics
print("=" * 60)
print("DESCRIPTIVE STATISTICS")
print("=" * 60)

# %%
# Summary statistics
print("\nSummary statistics:")
print(df.describe())

# %%
# Mean values
print("\nMean values:")
print(df[["age", "bmi", "children", "charges"]].mean())

# %%
# Median values
print("\nMedian values:")
print(df[["age", "bmi", "children", "charges"]].median())

# %%
# Minimum values
print("\nMinimum values:")
print(df[["age", "bmi", "children", "charges"]].min())

# %%
# Maximum values
print("\nMaximum values:")
print(df[["age", "bmi", "children", "charges"]].max())

# %%
print("=" * 60)
print("GROUP ANALYSIS")
print("=" * 60)

# %%
# Average charges by smoking status
print("\nAverage charges by smoking status:")
print(
    df.groupby("smoker")["charges"].mean()
)

# %%
# Average charges by region
print("\nAverage charges by region:")
print(
    df.groupby("region")["charges"].mean()
)

# %%
# Average charges by sex
print("\nAverage charges by sex:")
print(
    df.groupby("sex")["charges"].mean()
)

# %%
# Number of customers by region
print("\nNumber of customers in each region:")
print(
    df["region"].value_counts()
)

# %%
# Number of smokers and non-smokers
print("\nSmoking status:")
print(
    df["smoker"].value_counts()
)

# %%
# Distribution Of Medical Charges
plt.figure(figsize=(10, 6))

plt.hist(
    df["charges"],
    bins=30,
    edgecolor="black"
)

plt.title(
    "Distribution of Medical Insurance Charges"
)

plt.xlabel("Medical Charges")

plt.ylabel("Number of Customers")

plt.grid(axis="y", alpha=0.3)

plt.show()


# %%
# Charges by Smoking Status
plt.figure(figsize=(8, 6))

df.boxplot(
    column="charges",
    by="smoker"
)

plt.suptitle("")

plt.title(
    "Medical Charges by Smoking Status"
)

plt.xlabel("Smoking Status")

plt.ylabel("Medical Charges")

plt.grid(axis="y", alpha=0.3)

plt.show()

# %%
plt.figure(figsize=(10, 6))

plt.scatter(
    df["age"],
    df["charges"],
    alpha=0.5
)

plt.title(
    "Age vs Medical Insurance Charges"
)

plt.xlabel("Age")

plt.ylabel("Medical Charges")

plt.grid(alpha=0.3)

plt.show()

# %%
# BMI vs Medical Charges
plt.figure(figsize=(10, 6))

plt.scatter(
    df["bmi"],
    df["charges"],
    alpha=0.5
)

plt.title(
    "BMI vs Medical Insurance Charges"
)

plt.xlabel("BMI")

plt.ylabel("Medical Charges")

plt.grid(alpha=0.3)

plt.show()

# %%
# Average Charges by Region

average_charges_region = (
    df.groupby("region")["charges"]
    .mean()
)

plt.figure(figsize=(9, 6))

plt.bar(
    average_charges_region.index,
    average_charges_region.values,
    edgecolor="black"
)

plt.title(
    "Average Medical Charges by Region"
)

plt.xlabel("Region")

plt.ylabel("Average Medical Charges")

plt.xticks(rotation=20)

plt.grid(axis="y", alpha=0.3)

plt.show()

# %%
# Smokers vs Non-Smokers
smoking_counts = df["smoker"].value_counts()

plt.figure(figsize=(8, 8))

plt.pie(
    smoking_counts.values,
    labels=smoking_counts.index,
    autopct="%1.1f%%",
    startangle=90
)

plt.title(
    "Percentage of Smokers and Non-Smokers"
)

plt.show()

# %%
# Averge Charges by Age
average_charges_age = (
    df.groupby("age")["charges"]
    .mean()
)

plt.figure(figsize=(11, 6))

plt.plot(
    average_charges_age.index,
    average_charges_age.values,
    marker="o"
)

plt.title(
    "Average Medical Charges by Age"
)

plt.xlabel("Age")

plt.ylabel("Average Medical Charges")

plt.grid(alpha=0.3)

plt.show()

# %%
# Correlation Heatmap
correlation = df[
    [
        "age",
        "bmi",
        "children",
        "charges"
    ]
].corr()

plt.figure(figsize=(9, 7))

plt.imshow(
    correlation,
    interpolation="nearest"
)

plt.colorbar()

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=45
)

plt.yticks(
    range(len(correlation.columns)),
    correlation.columns
    )

plt.title(
    "Correlation Heatmap"
)

# Add correlation values
for i in range(len(correlation.columns)):
    for j in range(len(correlation.columns)):

        plt.text(
            j,
            i,
            f"{correlation.iloc[i, j]:.2f}",
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.show()

# %%

# PREPARE DATA FOR MACHINE LEARNING

print("=" * 60)
print("PREPARING DATA FOR MACHINE LEARNING")
print("=" * 60)





# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# %%
# Separate features and target
X = df.drop(columns=["charges"])

y = df["charges"]

print("\nOriginal features:")
print(X.columns.tolist())

print("\nTarget:")
print("charges")

# %%
# Convert categorical columns into numerical columns
X = pd.get_dummies(
    X,
    columns=["sex", "smoker", "region"],
    drop_first=True,
    dtype=int
)

print("\nFeatures after encoding:")
print(X.columns.tolist())

print("\nNumber of features:")
print(X.shape[1])

print("\nData preparation completed successfully!")

# %%

# TRAIN / TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training data:")
print(X_train.shape)

print("\nTesting data:")
print(X_test.shape)

print("\nTrain/test split completed!")

# %%
print("=" * 60)
print("DATA PREPROCESSING")
print("=" * 60)

# %%
# Create a copy of the training and testing data
X_train_processed = X_train.copy()
X_test_processed = X_test.copy()

# %%
# Scale the numerical features
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

numeric_features = [
    "age",
    "bmi",
    "children"
]

# %%
# Fit the scaler using the training data
X_train_processed[numeric_features] = scaler.fit_transform(
    X_train_processed[numeric_features]
)

# %%
# Transform the testing data
X_test_processed[numeric_features] = scaler.transform(
    X_test_processed[numeric_features]
)

print("\nNumerical features have been scaled successfully.")

print("\nTraining data shape:")
print(X_train_processed.shape)

print("\nTesting data shape:")
print(X_test_processed.shape)

print("\nData preprocessing completed successfully!")

# %%
# ============================================================
# TEST: SCIKIT-LEARN
# ============================================================

import sklearn

print("scikit-learn version:", sklearn.__version__)

from sklearn.linear_model import LinearRegression

print("LinearRegression works!")

# %%
# BATCH 17: LINEAR REGRESSION

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

print("=" * 60)
print("LINEAR REGRESSION MODEL")
print("=" * 60)



# %%

# Create the Linear Regression model
linear_model = LinearRegression()

# %%

# Train the model
linear_model.fit(
    X_train_processed,
    y_train
)

# %%
# Make predictions
linear_predictions = linear_model.predict(
    X_test_processed
)

# %%
# Calculate evaluation metrics
linear_mae = mean_absolute_error(
    y_test,
    linear_predictions
)

linear_mse = mean_squared_error(
    y_test,
    linear_predictions
)

linear_rmse = linear_mse ** 0.5

linear_r2 = r2_score(
    y_test,
    linear_predictions
)

# %%
# Display results
print("\nLinear Regression Results")
print("-" * 40)

print(f"Mean Absolute Error (MAE): {linear_mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {linear_rmse:.2f}")
print(f"R² Score: {linear_r2:.4f}")

print("\nLinear Regression completed successfully!")

# %%
# ============================================================
# BATCH 18: RIDGE REGRESSION
# ============================================================

from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

print("=" * 60)
print("RIDGE REGRESSION MODEL")
print("=" * 60)

# Create the Ridge Regression model
ridge_model = Ridge(alpha=1.0)

# Train the model
ridge_model.fit(
    X_train_processed,
    y_train
)

# Make predictions
ridge_predictions = ridge_model.predict(
    X_test_processed
)

# Calculate evaluation metrics
ridge_mae = mean_absolute_error(
    y_test,
    ridge_predictions
)

ridge_mse = mean_squared_error(
    y_test,
    ridge_predictions
)

ridge_rmse = ridge_mse ** 0.5

ridge_r2 = r2_score(
    y_test,
    ridge_predictions
)

# Display results
print("\nRidge Regression Results")
print("-" * 40)

print(f"Mean Absolute Error (MAE): {ridge_mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {ridge_rmse:.2f}")
print(f"R² Score: {ridge_r2:.4f}")

print("\nRidge Regression completed successfully!")

# %%
# RANDOM FOREST REGRESSION

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

print("=" * 60)
print("RANDOM FOREST REGRESSION MODEL")
print("=" * 60)

# Create the Random Forest model
random_forest_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

# Train the model
random_forest_model.fit(
    X_train_processed,
    y_train
)

# Make predictions
random_forest_predictions = random_forest_model.predict(
    X_test_processed
)

# Calculate evaluation metrics
rf_mae = mean_absolute_error(
    y_test,
    random_forest_predictions
)

rf_mse = mean_squared_error(
    y_test,
    random_forest_predictions
)

rf_rmse = rf_mse ** 0.5

rf_r2 = r2_score(
    y_test,
    random_forest_predictions
)

# Display results
print("\nRandom Forest Regression Results")
print("-" * 40)

print(f"Mean Absolute Error (MAE): {rf_mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rf_rmse:.2f}")
print(f"R² Score: {rf_r2:.4f}")

print("\nRandom Forest Regression completed successfully!")

# %%
# COMPARE MACHINE LEARNING MODELS

print("=" * 60)
print("MACHINE LEARNING MODEL COMPARISON")
print("=" * 60)

# Create a table comparing the three models
model_results = pd.DataFrame({

    "Model": [
        "Linear Regression",
        "Ridge Regression",
        "Random Forest"
    ],

    "MAE": [
        linear_mae,
        ridge_mae,
        rf_mae
    ],

    "RMSE": [
        linear_rmse,
        ridge_rmse,
        rf_rmse
    ],

    "R2": [
        linear_r2,
        ridge_r2,
        rf_r2
    ]
})

# Display the comparison table
print("\nModel Performance:")
print(model_results.to_string(index=False))

# Identify the best model using R²
best_model = model_results.loc[
    model_results["R2"].idxmax(),
    "Model"
]

best_r2 = model_results["R2"].max()

print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)

print(f"Best model: {best_model}")
print(f"Best R² Score: {best_r2:.4f}")

# %%
# BATCH 21: MODEL COMPARISON GRAPH

plt.figure(figsize=(10, 6))

plt.bar(
    model_results["Model"],
    model_results["R2"],
    edgecolor="black"
)

plt.title("Machine Learning Model Comparison")

plt.xlabel("Model")

plt.ylabel("R² Score")

plt.ylim(0, 1)

plt.xticks(rotation=20)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.show()

# %%
# BATCH 22: CROSS-VALIDATION

from sklearn.model_selection import KFold, cross_val_score

print("=" * 60)
print("5-FOLD CROSS-VALIDATION")
print("=" * 60)

# Create 5-fold cross-validation
kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Evaluate the Random Forest model
cv_scores = cross_val_score(
    random_forest_model,
    X,
    y,
    cv=kf,
    scoring="r2"
)

# Display each fold's R² score
print("\nCross-validation R² scores:")
print(cv_scores)

# Calculate the average score
average_cv_r2 = cv_scores.mean()

print(f"\nAverage Cross-validation R²: {average_cv_r2:.3f}")

print("\nCross-validation completed successfully!")

# %%
# BATCH 22: CROSS-VALIDATION

from sklearn.model_selection import KFold, cross_val_score

print("=" * 60)
print("5-FOLD CROSS-VALIDATION")
print("=" * 60)

# Create 5-fold cross-validation
kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Evaluate the Random Forest model
cv_scores = cross_val_score(
    random_forest_model,
    X,
    y,
    cv=kf,
    scoring="r2"
)

# Display each fold's R² score
print("\nCross-validation R² scores:")
print(cv_scores)

# Calculate the average score
average_cv_r2 = cv_scores.mean()

print(f"\nAverage Cross-validation R²: {average_cv_r2:.3f}")

print("\nCross-validation completed successfully!")

# %%
# BATCH 23: FEATURE IMPORTANCE

feature_importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance": random_forest_model.feature_importances_

})

# Sort from most important to least important
feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

print(feature_importance)

# %%
# FEATURE IMPORTANCE GRAPH


plt.figure(figsize=(10, 6))

plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"],
    edgecolor="black"
)

plt.title(
    "Random Forest Feature Importance"
)

plt.xlabel("Importance")

plt.ylabel("Feature")

plt.gca().invert_yaxis()

plt.grid(axis="x", alpha=0.3)

plt.tight_layout()

plt.show()

# %%
# BATCH 25: PREDICT MEDICAL INSURANCE CHARGES

# Create a sample customer
new_customer = pd.DataFrame({
    "age": [40],
    "sex": ["male"],
    "bmi": [30.5],
    "children": [2],
    "smoker": ["no"],
    "region": ["southeast"]
})

# Convert categorical variables to numbers
new_customer = pd.get_dummies(
    new_customer,
    columns=["sex", "smoker", "region"],
    drop_first=True,
    dtype=int
)

# Make sure the columns match the training data
new_customer = new_customer.reindex(
    columns=X.columns,
    fill_value=0
)

# Make prediction
predicted_charge = random_forest_model.predict(
    new_customer
)

print("=" * 60)
print("MEDICAL INSURANCE CHARGE PREDICTION")
print("=" * 60)

print(
    f"\nPredicted medical insurance charge: "
    f"${predicted_charge[0]:,.2f}"
)

# %%
# K-MEANS CLUSTERING

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Select variables for clustering
cluster_data = df[
    [
        "age",
        "bmi",
        "children",
        "charges"
    ]
]

# Standardise the data
cluster_scaler = StandardScaler()

cluster_scaled = cluster_scaler.fit_transform(
    cluster_data
)

# Create four customer clusters
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

# Assign clusters
df["cluster"] = kmeans.fit_predict(
    cluster_scaled
)

print("=" * 60)
print("K-MEANS CUSTOMER CLUSTERING")
print("=" * 60)

print("\nNumber of customers in each cluster:")
print(df["cluster"].value_counts().sort_index())

print("\nClustering completed successfully!")

# %%
# BATCH 27: CLUSTER SUMMARY

cluster_summary = df.groupby("cluster")[
    [
        "age",
        "bmi",
        "children",
        "charges"
    ]
].mean()

print("=" * 60)
print("CUSTOMER CLUSTER SUMMARY")
print("=" * 60)

print(cluster_summary)

# %%
# CLUSTER VISUALISATION
# Age vs Medical Charges

plt.figure(figsize=(10, 6))

for cluster_number in sorted(df["cluster"].unique()):

    cluster_data_plot = df[
        df["cluster"] == cluster_number
    ]

    plt.scatter(
        cluster_data_plot["age"],
        cluster_data_plot["charges"],
        alpha=0.6,
        label=f"Cluster {cluster_number}"
    )

plt.title(
    "Customer Clusters: Age vs Medical Charges"
)

plt.xlabel("Age")

plt.ylabel("Medical Charges")

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

plt.show()

# %%
# BATCH 29: AVERAGE CHARGES BY CUSTOMER CLUSTER
average_cluster_charges = df.groupby(
    "cluster"
)["charges"].mean()

plt.figure(figsize=(9, 6))

plt.bar(
    average_cluster_charges.index.astype(str),
    average_cluster_charges.values,
    edgecolor="black"
)

plt.title(
    "Average Medical Charges by Customer Cluster"
)

plt.xlabel("Customer Cluster")

plt.ylabel("Average Medical Charges")

plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.show()

# %%
# FINAL MODEL SUMMARY

print("=" * 60)
print("FINAL MACHINE LEARNING SUMMARY")
print("=" * 60)

print("\nMODEL PERFORMANCE:")
print(model_results)

print("\nBEST MODEL:")
best_model = model_results.loc[
    model_results["R2"].idxmax()
]

print(best_model)

print("\nCROSS-VALIDATION:")
print(
    f"Average 5-Fold Cross-validation R²: "
    f"{cv_scores.mean():.3f}"
)

print("\nCUSTOMER CLUSTERS:")
print(
    f"Number of clusters: "
    f"{df['cluster'].nunique()}"
)

print("\nPROJECT COMPLETED SUCCESSFULLY!")

# %%
# FINAL PROJECT CONCLUSION

print("=" * 60)
print("FINAL PROJECT CONCLUSION")
print("=" * 60)

print("""
The medical insurance dataset was successfully analysed
using Python, data visualisation and machine learning.

The analysis examined customer characteristics such as age,
BMI, smoking status, number of children and region.

Three machine learning models were compared:
1. Linear Regression
2. Ridge Regression
3. Random Forest

Random Forest achieved the strongest predictive performance.

Five-fold cross-validation produced an average R² score
of approximately 0.982, showing that the model performed
consistently across different subsets of the data.

Feature importance analysis was used to identify the
variables that contributed most to predicting medical
insurance charges.

K-Means clustering was also applied to identify groups
of customers with similar characteristics.

Overall, the project demonstrates how data analytics,
visualisation and machine learning can be used to understand
medical insurance costs and support data-driven decision making.
""")

print("\nPROJECT ANALYSIS COMPLETE!")

# %%
# SAVE PROJECT RESULTS

# Save model comparison
model_results.to_csv(
    "results/model_comparison_results.csv",
    index=False
)

# Save feature importance
feature_importance.to_csv(
    "results/feature_importance_results.csv",
    index=False
)

# Save cluster summary
cluster_summary.to_csv(
    "results/cluster_summary_results.csv"
)

# Save the cleaned dataset with clusters
df.to_csv(
    "Results/insurance_analysis_results.csv",
    index=False
)

print("=" * 60)
print("RESULTS SAVED SUCCESSFULLY")
print("=" * 60)

print("\nFiles created:")
print("- model_comparison_results.csv")
print("- feature_importance_results.csv")
print("- cluster_summary_results.csv")
print("- insurance_analysis_results.csv")

# %%



