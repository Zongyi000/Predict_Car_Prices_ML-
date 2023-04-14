import numpy as np
import pandas as pd

import random
from car_predict_model import ANNPredictionModel
import matplotlib.pyplot as plt


def random_choice_another_value(df, column, unique_values):
    random_value = random.choice(unique_values)
    while random_value == df[column]:
        random_value = random.choice(unique_values)
    return random_value


def generate_10_scaled_values(df, column, min_value, max_value):
    result = []
    for i in range(10):
        result.append(min_value + (max_value - min_value) * i / 10)
    return result


def main():
    df = pd.read_csv('dataset/test/test.csv')
    unique_manufacturers = df['manufacturer'].unique()
    unique_year = df['year'].unique()
    unique_fuel = df['fuel'].unique()
    unique_title_status = df['title_status'].unique()
    unique_transmission = df['transmission'].unique()
    unique_type = df['type'].unique()
    unique_state = df['state'].unique()
    min_odometer = df['odometer'].min()
    max_odometer = df['odometer'].max()
    print(min_odometer, max_odometer)
    first_50 = df[:50]
    model = ANNPredictionModel('ann/model/ann_best.ckpt')
    category_columns_unique_values = {'year': unique_year, 'manufacturer': unique_manufacturers, 'fuel': unique_fuel,
                                      'title_status': unique_title_status, 'transmission': unique_transmission,
                                      'type': unique_type, 'state': unique_state}
    numerical_columns = ['odometer']

    result = []
    original_predictions = []
    for index, row in first_50.iterrows():
        original_predictions.append(model.predict(pd.DataFrame([row])))

    for column in first_50.columns:
        if column == 'price':
            continue
        result.append([])
        if column in category_columns_unique_values:
            print(column)
            for index, row in first_50.iterrows():
                plt.scatter(row[column], original_predictions[index], c='r', marker='o')
                result[-1].append([])
                for i in range(10):
                    new_row = row.copy()
                    new_row[column] = random_choice_another_value(new_row, column,
                                                                  category_columns_unique_values[column])
                    new_value = new_row[column]
                    new_row = pd.DataFrame([new_row], columns=df.columns)
                    new_prediction = model.predict(new_row)
                    print(new_prediction)
                    result[-1][-1].append(new_prediction)
                    plt.scatter(new_value, new_prediction, c='b', marker='o')
        elif column in numerical_columns:
            print(column)
            for index, row in first_50.iterrows():
                plt.scatter(row[column], original_predictions[index], c = 'r', marker='o')
                result[-1].append([])
                new_values = generate_10_scaled_values(row, column, min_odometer, max_odometer)
                for new_value in new_values:
                    new_row = row.copy()
                    new_row[column] = new_value
                    new_row = pd.DataFrame([new_row], columns=df.columns)
                    # print(new_row)
                    new_prediction = model.predict(new_row)
                    # print(new_prediction)
                    result[-1][-1].append(new_prediction)
                    plt.scatter(new_value, new_prediction, c='b', marker='o')
        plt.xlabel(column)
        plt.ylabel('price')
        plt.legend(['original', 'changed'])
        plt.show()
        # save figure
        plt.savefig('sensitive_analysis/' + column + '.png')

    # calculate std dev
    change_map = {}
    for i, column in enumerate(first_50.columns):
        if column == 'price':
            continue
        print(i)
        # flatten result[i]
        result[i] = [item for sublist in result[i] for item in sublist]
        new_prediction = np.array(result[i])
        # rmse of result[i] to original_predictions[i]
        original_prediction = original_predictions[i] * np.ones(len(result[i]))
        change_map[column] = np.sqrt(np.mean((new_prediction - original_prediction) ** 2))
    print(change_map)
    # convert result to csv
    result = pd.DataFrame(result)
    result.to_csv('sensitive_analysis/result.csv')


if __name__ == '__main__':
    main()
