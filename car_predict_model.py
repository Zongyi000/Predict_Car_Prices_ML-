from abc import abstractmethod
import torch
import joblib

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
