from abc import abstractmethod
import joblib
import pandas as pd


class CarPredictionModel:

    @abstractmethod
    def __init__(self, model_path):
        pass

    @abstractmethod
    def predict(self, X) -> float:
        pass


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