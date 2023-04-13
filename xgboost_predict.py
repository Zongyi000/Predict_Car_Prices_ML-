from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from joblib import dump, load
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb


def main():
    # Step 1: Load the dataset
    car_data = pd.read_csv('car_data.csv')

    # Step 2: Encode label
    le = LabelEncoder()
    print(car_data.columns)
    car_data[['Car_Name', 'Year', 'Selling_Price', 'Present_Price', 'Kms_Driven',
              'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']] = car_data[
        ['Car_Name', 'Year', 'Selling_Price', 'Present_Price', 'Kms_Driven',
         'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']].apply(le.fit_transform)
    print(car_data)

    # Step 3: Prepare the data
    X = car_data.drop(['Selling_Price'], axis=1)
    y = car_data['Selling_Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Step 4: Train the model
    xgb_model = xgb.XGBRegressor(
        n_estimators=120,  # number of trees
        learning_rate=0.01,  # step size shrinkage
        max_depth=7,  # maximum depth of each tree
        reg_lambda=0.5,  # L2 regularization term
        objective='reg:squarederror'  # loss function
    )
    xgb_model.fit(X_train, y_train)

    # Step 5: Evaluate the model
    y_pred = xgb_model.predict(X_test)
    print('R-squared:', r2_score(y_test, y_pred))
    print('MAE:', mean_absolute_error(y_test, y_pred))
    print('RMSE:', mean_squared_error(y_test, y_pred, squared=False))

    # Step 6: Tune the hyperparameters
    param_grid = {
        'n_estimators': [60, 100, 120, 140],
        'learning_rate': [0.01, 0.1],
        'max_depth': [5, 7],
        'reg_lambda': [0.5]
    }
    xgb_reg = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, n_jobs=-1).fit(X_train, y_train)
    print("Best score: %0.3f" % xgb_reg.best_score_)
    print("Best parameters set:", xgb_reg.best_params_)

    # Step 7: Use the best model parameters
    xgb_model_2 = xgb.XGBRegressor(
        n_estimators=100,  # number of trees
        learning_rate=0.1,  # step size shrinkage
        max_depth=5,  # maximum depth of each tree
        reg_lambda=0.5,  # L2 regularization term
        objective='reg:squarederror'  # loss function
    )
    xgb_model_2.fit(X_train, y_train)

    # Step 8: Compare result
    y_pred_2 = xgb_model_2.predict(X_test)
    print('Best Model R-squared:', r2_score(y_test, y_pred_2))
    print('Best Model MAE:', mean_absolute_error(y_test, y_pred_2))
    print('Best Model RMSE:', mean_squared_error(y_test, y_pred_2, squared=False))

    # Step 9: Evaluate feature importance
    xgb.plot_importance(xgb_model_2)
    plt.tight_layout()
    plt.show()

    # Step 10: Save the model
    dump(xgb_reg, "xgb_model.joblib")


if __name__ == "__main__":
    main()
