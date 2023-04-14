from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OrdinalEncoder
from joblib import dump, load
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb


def main():
    # Step 1: Load the dataset
    car_data = pd.read_csv('../dataset/train/train.csv')
    car_data_test = pd.read_csv('../dataset/test/test.csv')
    print(car_data)
    columns = car_data.columns

    # Step 2: Encode label
    numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    categorical_columns = []
    features = car_data.columns.values.tolist()
    for col in features:
        if car_data[col].dtype in numerics: continue
        categorical_columns.append(col)

    # Step 3: Concatenate train and test data
    combined_data = pd.concat([car_data, car_data_test], axis=0)

    # Step 4: Transform and save preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('ordinal', OrdinalEncoder(),
             ['manufacturer', 'fuel', 'title_status', 'transmission', 'type', 'state'])
        ],
        remainder='passthrough'
    )
    combined_data = preprocessor.fit_transform(combined_data)
    dump(preprocessor, 'preprocessor.joblib')

    # Step 5: Separate dataset and set year odometer value
    columns = ['manufacturer', 'fuel', 'title_status', 'transmission', 'type', 'state', 'year', 'odometer', 'price']
    combined_data = pd.DataFrame(combined_data, columns=columns)
    print(combined_data)

    car_data = combined_data[:len(car_data)]
    car_data_test = combined_data[len(car_data):]

    car_data.loc[:, 'year'] = (car_data['year'] - 1900).astype(int)
    car_data.loc[:, 'odometer'] = car_data['odometer'].astype(int)
    car_data_test.loc[:, 'year'] = (car_data_test['year'] - 1900).astype(int)
    car_data_test.loc[:, 'odometer'] = car_data_test['odometer'].astype(int)
    print(car_data)

    # Step 6: Prepare the train and test dataset
    X = car_data.drop(['price'], axis=1)
    y = car_data['price']
    X_test = car_data_test.drop(['price'], axis=1)
    y_test = car_data_test['price']
    X_train = X
    y_train = y

    # Step 7: Tune the hyperparameters
    xgb_clf = xgb.XGBRegressor(objective='reg:squarederror')
    param_grid = {
        'n_estimators': [60, 100, 120, 140],
        'learning_rate': [0.01, 0.1],
        'max_depth': [5, 7],
        'reg_lambda': [0.5]
    }
    xgb_reg = GridSearchCV(estimator=xgb_clf, param_grid=param_grid, cv=5, n_jobs=-1).fit(X_train, y_train)
    print("Best score: %0.3f" % xgb_reg.best_score_)
    print("Best parameters set:", xgb_reg.best_params_)

    # Step 8: Use the best model parameters
    xgb_model_best = xgb.XGBRegressor(
        n_estimators=140,  # number of trees
        learning_rate=0.1,  # step size shrinkage
        max_depth=7,  # maximum depth of each tree
        reg_lambda=0.5,  # L2 regularization term
        objective='reg:squarederror'  # loss function
    )
    xgb_model_best.fit(X_train, y_train)

    # Step 9: Evaluate test dataset
    y_pred_best = xgb_model_best.predict(X_test)
    print('Best Model R-squared:', r2_score(y_test, y_pred_best))
    print('Best Model MAE:', mean_absolute_error(y_test, y_pred_best))
    print('Best Model RMSE:', mean_squared_error(y_test, y_pred_best, squared=False))

    # Step 10: Evaluate feature importance
    xgb.plot_importance(xgb_model_best)
    plt.tight_layout()
    plt.show()

    # Step 11: Save the XGBoost model
    dump(xgb_model_best, "xgb_model.joblib")


if __name__ == "__main__":
    main()
