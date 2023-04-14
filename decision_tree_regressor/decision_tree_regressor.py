import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from joblib import dump
from math import sqrt


def main():
    # 0run preprocess.py to filter the dataset and get train.csv in
    # 1Load the dataset
    data = pd.read_csv('../dataset/train/train.csv')

    # 2Encode categorical features
    encoder = LabelEncoder()
    categorical_features = ["manufacturer", "fuel", "title_status", "transmission", "type", "state"]
    for feature in categorical_features:
        data[feature] = encoder.fit_transform(data[feature])

    # 3Preprocess data
    X = data.drop("price", axis=1)
    y = data["price"]
    # print("step 3:")
    # print(X)

    # 4Scale numerical features:standardizes the numerical features in the dataset
    scaler = StandardScaler()
    numerical_features = ["year", "odometer"]
    X[numerical_features] = scaler.fit_transform(X[numerical_features])
    # print("step 4:")
    # print(X)

    # 5Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 6Train the model:find the best hyperparameters for the DecisionTreeRegressor model
    regressor = DecisionTreeRegressor(random_state=42)
    param_grid = {
        "max_depth": [None, 10, 20, 30, 40, 50],
        "min_samples_split": [2, 10, 20, 30, 40],
        "min_samples_leaf": [1, 5, 10, 20, 30]
    }

    grid_search = GridSearchCV(estimator=regressor, param_grid=param_grid, scoring="neg_mean_squared_error", cv=5,
                               n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # 7Evaluate the model
    best_regressor = grid_search.best_estimator_
    y_pred = best_regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean squared error: {mse:.2f}")
    print(f"RMSE: {sqrt(mse):.2f}")
    print(f"R^2 score: {r2:.2f}")

    # 8Evaluate feature importance
    feature_importance = best_regressor.feature_importances_
    for feature, importance in zip(X.columns, feature_importance):
        print("step 10: feature importance below:")
        print(f"{feature}: {importance:.4f}")
        print("-----")

    # 9Save the model
    dump(best_regressor, "decision_tree_regressor.joblib")


if __name__ == "__main__":
    main()


# Mean squared error: 31985992.80 and R^2 score: 0.80,
# feature importance below: year: 0.5639    type: 0.1236   odometer: 0.1103    manufacturer: 0.0884   fuel: 0.0792  state: 0.0108  transmission: 0.0172   title_status: 0.0067


# MSE is too high, One way to potentially improve the model's performance is to try a different algorithm, such as the RandomForestRegressor, which is an ensemble method and usually performs better than a single decision tree.
# step 5 may modified into:
# regressor = RandomForestRegressor(random_state=42)
# param_grid = {
#     "n_estimators": [50, 100, 200],
#     "max_depth": [None, 10, 20, 30],
#     "min_samples_split": [2, 10, 20],
#     "min_samples_leaf": [1, 5, 10]
# }
