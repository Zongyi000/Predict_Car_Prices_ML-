import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from joblib import dump
from math import sqrt


def main():
    # 0run preprocess.py to filter the dataset and get train.csv in dataset
    # Load the dataset
    data = pd.read_csv('../dataset/train/train.csv')

    # Encode categorical features
    encoder = LabelEncoder()
    categorical_features = ["manufacturer", "fuel", "title_status", "transmission", "type", "state"]
    for feature in categorical_features:
        data[feature] = encoder.fit_transform(data[feature])

    # Preprocess data
    X = data.drop("price", axis=1)
    y = data["price"]

    # Scale numerical features
    scaler = StandardScaler()
    numerical_features = ["year", "odometer"]
    X[numerical_features] = scaler.fit_transform(X[numerical_features])

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    regressor = RandomForestRegressor(random_state=42)
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 10, 20],
        "min_samples_leaf": [1, 5, 10]
    }

    # to accelerate, use RandomizedSearchCV instead of GridSearchCV and change cv to 3
    random_search = RandomizedSearchCV(estimator=regressor, param_distributions=param_grid, n_iter=20, scoring="neg_mean_squared_error", cv=3, n_jobs=-1, random_state=42)
    random_search.fit(X_train, y_train)

    # Evaluate the model
    # best_regressor = grid_search.best_estimator_
    best_regressor = random_search.best_estimator_
    # Make predictions using the model
    y_pred = best_regressor.predict(X_test)

    # Calculate MSE and R^2
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean squared error: {mse:.2f}")
    print(f"RMSE: {sqrt(mse):.2f}")
    print(f"R^2 score: {r2:.2f}")

    # Compare predicted and actual prices
    print("Predicted Prices vs. Actual Prices")
    for pred, actual in zip(y_pred[:10], y_test[:10]):
        print(f"Predicted: {pred:.2f} | Actual: {actual:.2f}")

    # Evaluate feature importance
    feature_importance = best_regressor.feature_importances_
    print("feature importance below:")
    for feature, importance in zip(X.columns, feature_importance):
        print(f"{feature}: {importance:.4f}")
    print("-----")

    # Sensitivity analysis
    sensitivity_analysis = {}
    for feature in X.columns:
        X_test_copy = X_test.copy()
        X_test_copy[feature] *= 1.05
        y_pred_copy = best_regressor.predict(X_test_copy)
        mse_copy = mean_squared_error(y_test, y_pred_copy)
        sensitivity_analysis[feature] = (mse_copy - mse) / mse

    # Print sensitivity analysis results
    print("\nSensitivity Analysis:")
    for feature, sensitivity in sensitivity_analysis.items():
        print(f"{feature}: {sensitivity:.4f}")

    # Inverse transform the standardized values
    X_test[numerical_features] = scaler.inverse_transform(X_test[numerical_features])

    # Save the model
    dump(best_regressor, "random_forest_regressor.joblib")


if __name__ == "__main__":
    main()


# Mean squared error: 24928059.49
# RMSE: 4992.80
# R^2 score: 0.84
# Predicted Prices vs. Actual Prices
# Predicted: 6534.49 | Actual: 7988.00
# Predicted: 12847.77 | Actual: 18500.00
# Predicted: 19079.07 | Actual: 15900.00
# Predicted: 6503.47 | Actual: 8200.00
# Predicted: 7783.74 | Actual: 5900.00
# Predicted: 23118.77 | Actual: 22590.00
# Predicted: 15063.33 | Actual: 18990.00
# Predicted: 5349.10 | Actual: 4500.00
# Predicted: 34561.31 | Actual: 20500.00
# Predicted: 20230.15 | Actual: 17990.00
# feature importance below:
# year: 0.5121
# manufacturer: 0.0929
# fuel: 0.0743
# odometer: 0.1444
# title_status: 0.0083
# transmission: 0.0190
# type: 0.1153
# state: 0.0337
# -----
#
# Sensitivity Analysis:
# year: 0.0031
# manufacturer: 0.5957
# fuel: 0.0002
# odometer: 0.2126
# title_status: -0.0001
# transmission: -0.0002
# type: 0.0265
# state: 0.0228


# The code runs slowly because it is performing an extensive search for the best hyperparameters using GridSearchCV. GridSearchCV exhaustively tries all combinations of the provided hyperparameters and evaluates them using cross-validation, which can be computationally expensive, especially when working with a large dataset or many parameter combinations.
# In the code provided, there are 3 values for n_estimators, 4 values for max_depth, 3 values for min_samples_split, and 3 values for min_samples_leaf. This results in 3 x 4 x 3 x 3 = 108 combinations of hyperparameters. Moreover, the cv=5 parameter in the GridSearchCV function performs 5-fold cross-validation, meaning that the model is trained and evaluated 108 x 5 = 540 times in total.
# To speed up the code, you can try the following approaches:
# Reduce the number of parameter combinations. For example, you can remove some values from the parameter grid or use fewer parameters.
# Use a smaller value for cv in the GridSearchCV function to reduce the number of cross-validation folds.
# Use a different hyperparameter tuning method, such as RandomizedSearchCV, which samples a random subset of parameter combinations instead of searching exhaustively. This can provide a good balance between computational cost and finding optimal hyperparameters.