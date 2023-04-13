from sklearn.model_selection import GridSearchCV
from joblib import dump, load
import xgboost as xgb
import pandas as pd

processed_dataset = "car_data.csv"


train0 = pd.read_csv('/kaggle/input/craigslist-carstrucks-data/craigslistVehicles.csv')
target_name = 'price'
train_target0 = train0[target_name]
train0 = train0.drop([target_name], axis=1)

# Synthesis test0 from train0
train0, test0, train_target0, test_target0 = train_test_split(train0, train_target0, test_size=0.2, random_state=0)

# For boosting model
train0b = train0
train_target0b = train_target0
# Synthesis valid as test for selection models
trainb, testb, targetb, target_testb = train_test_split(train0b, train_target0b, test_size=valid_part, random_state=0)


def acc_boosting_model(num, model, train, test, num_iteration=0):
    # Calculation of accuracy of boosting model by different metrics

    global acc_train_r2, acc_test_r2, acc_train_d, acc_test_d, acc_train_rmse, acc_test_rmse

    if num_iteration > 0:
        ytrain = model.predict(train, num_iteration=num_iteration)
        ytest = model.predict(test, num_iteration=num_iteration)
    else:
        ytrain = model.predict(train)
        ytest = model.predict(test)

    print('target = ', targetb[:5].values)
    print('ytrain = ', ytrain[:5])

    acc_train_r2_num = round(r2_score(targetb, ytrain) * 100, 2)
    print('acc(r2_score) for train =', acc_train_r2_num)
    acc_train_r2.insert(num, acc_train_r2_num)

    acc_train_d_num = round(acc_d(targetb, ytrain) * 100, 2)
    print('acc(relative error) for train =', acc_train_d_num)
    acc_train_d.insert(num, acc_train_d_num)

    acc_train_rmse_num = round(acc_rmse(targetb, ytrain) * 100, 2)
    print('acc(rmse) for train =', acc_train_rmse_num)
    acc_train_rmse.insert(num, acc_train_rmse_num)

    print('target_test =', target_testb[:5].values)
    print('ytest =', ytest[:5])

    acc_test_r2_num = round(r2_score(target_testb, ytest) * 100, 2)
    print('acc(r2_score) for test =', acc_test_r2_num)
    acc_test_r2.insert(num, acc_test_r2_num)

    acc_test_d_num = round(acc_d(target_testb, ytest) * 100, 2)
    print('acc(relative error) for test =', acc_test_d_num)
    acc_test_d.insert(num, acc_test_d_num)

    acc_test_rmse_num = round(acc_rmse(target_testb, ytest) * 100, 2)
    print('acc(rmse) for test =', acc_test_rmse_num)
    acc_test_rmse.insert(num, acc_test_rmse_num)


def test():
    model = xgb.XGBRegressor({'objective': 'reg:squarederror'})
    parameters = {'n_estimators': [60, 100, 120, 140],
                  'learning_rate': [0.01, 0.1],
                  'max_depth': [5, 7],
                  'reg_lambda': [0.5]}
    xgb_reg = GridSearchCV(estimator=model, param_grid=parameters, cv=5, n_jobs=-1).fit(trainb, targetb)
    print("Best score: %0.3f" % xgb_reg.best_score_)
    print("Best parameters set:", xgb_reg.best_params_)
    acc_boosting_model(7, xgb_reg, trainb, testb)
    # dump(xgb_reg, "ttt")
