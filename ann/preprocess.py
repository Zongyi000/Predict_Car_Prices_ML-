import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import joblib


def preprocess_data(train_data):
    # Define the column transformer with OneHotEncoder for categorical features and StandardScaler for continuous features
    preprocessor = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False),
             ['manufacturer', 'fuel', 'title_status', 'transmission', 'type', 'state']),
            ('scaler', StandardScaler(), ['year', 'odometer']),
        ])

    # Fit and transform the training data
    X_train = preprocessor.fit_transform(train_data.drop('price', axis=1))
    y_train = train_data['price'].values
    joblib.dump(preprocessor, 'preprocessor.joblib')

    # split the data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

    return X_train, y_train, X_val, y_val


class CarPriceDataset(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        row = self.data[index]
        features = torch.tensor(row, dtype=torch.float)
        label = torch.tensor(self.target[index], dtype=torch.float)
        return features, label


def prepare_dataset():
    # Load the train and test CSV data
    train_data = pd.read_csv('../dataset/train/train.csv')

    # Preprocess the train and val data
    X_train, y_train, X_val, y_val = preprocess_data(train_data)

    price_scaler = StandardScaler()
    y_train_scaled = price_scaler.fit_transform(y_train.reshape(-1, 1)).reshape(-1)
    y_val_scaled = price_scaler.transform(y_val.reshape(-1, 1)).reshape(-1)
    joblib.dump(price_scaler, 'price_scaler.joblib')

    train_dataset = CarPriceDataset(X_train,
                                    y_train_scaled)
    val_dataset = CarPriceDataset(X_val,
                                  y_val_scaled)
    return train_dataset, val_dataset


def preprocess_test_data(test_data):
    # Load the preprocessor
    x_preprocessor = joblib.load('preprocessor.joblib')

    # Transform the test data
    X_test = x_preprocessor.transform(test_data.drop('price', axis=1))
    y_test = test_data['price'].values
    return X_test, y_test


def prepare_test_dataset():
    # Load the train and test CSV data
    test_data = pd.read_csv('../dataset/test/test.csv')

    # Preprocess the train and val data
    X_test, y_test = preprocess_test_data(test_data)

    test_dataset = CarPriceDataset(X_test, y_test)
    return test_dataset
