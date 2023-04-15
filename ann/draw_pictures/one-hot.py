import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt


def print_type(df):
    # Load the preprocessor
    encoder = OneHotEncoder(sparse=False)
    encoded_feature = encoder.fit_transform(df[['type']])
    # Transform the test data
    # Get the list of unique categories from the feature
    categories = encoder.categories_[0]

    # Create column names for the one-hot encoded features
    column_names = [f"{feature}_{category}" for feature, category in zip(['Feature'] * len(categories), categories)]

    # Create a DataFrame containing the one-hot encoded values
    encoded_df = pd.DataFrame(encoded_feature, columns=column_names)
    print(encoded_df[:5])

def print_Odometer(df):
    scaler = StandardScaler()
    # Transform the test data
    odometer_scaled = scaler.fit_transform(df[['odometer']])
    print(odometer_scaled)
    # draw distribution of odometer
    plt.figure(figsize=(10, 5))
    plt.hist(odometer_scaled, range=(-3, 3), bins=100)
    plt.xlabel('Odometer')
    plt.ylabel('Count')
    plt.xlim(-3, 3)
    plt.savefig('odometer.png')
    plt.show()

def prepare_test_dataset():
    # Load the train and test CSV data
    test_data = pd.read_csv('../../dataset/test/test.csv')
    # print_type(test_data)
    print_Odometer(test_data)
    return


prepare_test_dataset()
