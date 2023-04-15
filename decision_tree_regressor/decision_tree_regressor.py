import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from joblib import dump
from math import sqrt
import matplotlib.pyplot as plt


def plot_predictions(y_true, y_pred, title):
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title(title)
    plt.show()


def main():
    # 0run preprocess.py to filter the dataset and get train.csv in dataset
    # Load the dataset
    data = pd.read_csv('../dataset/train/train.csv')
    # data = pd.read_csv('../dataset/train/train.csv')

    # Encode categorical features
    encoders = {}
    categorical_features = ["manufacturer", "fuel", "title_status", "transmission", "type", "state"]
    for feature in categorical_features:
        encoders[feature] = LabelEncoder()
        data[feature] = encoders[feature].fit_transform(data[feature])
        # save the encoder
        joblib.dump(encoders[feature], f"{feature}_encoder.joblib")

    # Preprocess data
    X = data.drop("price", axis=1)
    y = data["price"]
    # print("step 3:")
    # print(X)

    # Scale numerical features:standardizes the numerical features in the dataset
    scaler = StandardScaler()
    numerical_features = ["year", "odometer"]
    X[numerical_features] = scaler.fit_transform(X[numerical_features])
    # print("step 4:")
    # print(X)

    joblib.dump(scaler, "scaler.joblib")


    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model:find the best hyperparameters for the DecisionTreeRegressor model
    regressor = DecisionTreeRegressor(random_state=42)
    param_grid = {
        "max_depth": [None, 10, 20, 30, 40, 50],
        "min_samples_split": [2, 10, 20, 30, 40],
        "min_samples_leaf": [1, 5, 10, 20, 30]
    }

    grid_search = GridSearchCV(estimator=regressor, param_grid=param_grid, scoring="neg_mean_squared_error", cv=5,
                               n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Evaluate the model
    best_regressor = grid_search.best_estimator_
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

    # Plot predictions for train set
    y_train_pred = best_regressor.predict(X_train)
    plot_predictions(y_train, y_train_pred, "Train Set: True vs. Predicted Prices")

    # Plot predictions for test set
    plot_predictions(y_test, y_pred, "Test Set: True vs. Predicted Prices")

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
    dump(best_regressor, "decision_tree_regressor.joblib")


if __name__ == "__main__":
    main()


# Mean squared error: 31985992.80
# RMSE: 5655.62
# R^2 score: 0.80
# Predicted Prices vs. Actual Prices
# Predicted: 6274.66 | Actual: 7988.00
# Predicted: 10465.26 | Actual: 18500.00
# Predicted: 21086.67 | Actual: 15900.00
# Predicted: 6744.70 | Actual: 8200.00
# Predicted: 7350.24 | Actual: 5900.00
# Predicted: 23065.00 | Actual: 22590.00
# Predicted: 15371.93 | Actual: 18990.00
# Predicted: 5905.00 | Actual: 4500.00
# Predicted: 34352.54 | Actual: 20500.00
# Predicted: 20523.33 | Actual: 17990.00
# feature importance below:
# year: 0.5639
# manufacturer: 0.0884
# fuel: 0.0792
# odometer: 0.1103
# title_status: 0.0067
# transmission: 0.0172
# type: 0.1236
# state: 0.0108
# -----
# Sensitivity Analysis:
# year: 0.0015
# manufacturer: 0.5070
# fuel: 0.0000
# odometer: 0.1913
# title_status: 0.0000
# transmission: -0.0003
# type: 0.0247
# state: 0.0067




# MSE is too high, One way to potentially improve the model's performance is to try a different algorithm, such as the RandomForestRegressor, which is an ensemble method and usually performs better than a single decision tree.
# step 5 may modified into:
# regressor = RandomForestRegressor(random_state=42)
# param_grid = {
#     "n_estimators": [50, 100, 200],
#     "max_depth": [None, 10, 20, 30],
#     "min_samples_split": [2, 10, 20],
#     "min_samples_leaf": [1, 5, 10]
# }
