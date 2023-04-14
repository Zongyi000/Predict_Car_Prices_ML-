from abc import abstractmethod
import torch
import joblib
import pandas as pd
from ann.model.ANN import ANN


class CarPredictionModel:

    @abstractmethod
    def __init__(self, model_path):
        pass

    @abstractmethod
    def predict(self, X) -> float:
        pass


class ANNPredictionModel(CarPredictionModel):
    def __init__(self, model_path):
        self.model = ANN(120, hidden_size=1024, output_size=1)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def predict(self, X) -> float:
        preprocessor = joblib.load("ann/preprocessor.joblib")
        price_scaler = joblib.load("ann/price_scaler.joblib")
        X = preprocessor.transform(X)
        X = torch.tensor(X, dtype=torch.float)  # Convert DataFrame to NumPy array and then to tensor
        y = self.model(X)
        y = y.detach().numpy()
        y = price_scaler.inverse_transform(y)
        return y[0][0]
class XGBoost(CarPredictionModel):
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        self.encoder = joblib.load("xgboost/preprocessor.joblib")

    def predict(self, X) -> float:
        # create a copy of the input data
        X_copy = X.copy()
        X_copy["price"] = 0
        X_copy["year"] -= 1900
        X_copy = X_copy.reindex(
            columns=['manufacturer', 'fuel', 'title_status', 'transmission', 'type', 'state', 'year', 'odometer', 'price'])
        # print(X_copy)

        # Transform the input data using the encoder (preprocessor)
        X_encoded = self.encoder.transform(X_copy)

        # Convert the transformed data to a DataFrame
        transformed_columns = X_copy.columns
        X_encoded_df = pd.DataFrame(X_encoded, columns=transformed_columns)
        # print(X_encoded_df)
        X_encoded_df.drop("price", axis=1, inplace=True)
        # print(X_encoded_df)

        # make predictions using the XGBoost model
        y_pred = self.model.predict(X_encoded_df)
        return y_pred
