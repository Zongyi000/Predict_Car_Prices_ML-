import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from joblib import load
from math import sqrt


def main():
    # Load the saved model
    loaded_regressor = load("./random_forest_regressor.joblib")

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

    # Evaluate feature importance
    feature_importance = loaded_regressor.feature_importances_
    for feature, importance in zip(X.columns, feature_importance):
        print(f"{feature}: {importance:.4f}")


if __name__ == "__main__":
    main()
