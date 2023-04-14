import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from joblib import load
from math import sqrt


def main():
    # run the preprocess.py to filter data
    # Load the saved model
    loaded_regressor = load("./decision_tree_regressor.joblib")

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

    # Make predictions using the loaded model
    y_pred = loaded_regressor.predict(X_test)

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
    feature_importance = loaded_regressor.feature_importances_
    print("feature importance below:")
    for feature, importance in zip(X.columns, feature_importance):
        print(f"{feature}: {importance:.4f}")
    print("-----")

    # Sensitivity analysis
    sensitivity_analysis = {}
    for feature in X.columns:
        X_test_copy = X_test.copy()
        X_test_copy[feature] *= 1.05
        y_pred_copy = loaded_regressor.predict(X_test_copy)
        mse_copy = mean_squared_error(y_test, y_pred_copy)
        sensitivity_analysis[feature] = (mse_copy - mse) / mse

    # Print sensitivity analysis results
    print("\nSensitivity Analysis:")
    for feature, sensitivity in sensitivity_analysis.items():
        print(f"{feature}: {sensitivity:.4f}")

    # Inverse transform the standardized values
    X_test[numerical_features] = scaler.inverse_transform(X_test[numerical_features])


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
#
# Sensitivity Analysis:
# year: 0.0015
# manufacturer: 0.5070
# fuel: 0.0000
# odometer: 0.1913
# title_status: 0.0000
# transmission: -0.0003
# type: 0.0247
# state: 0.0067

