from abc import abstractmethod


class CarPredictionModel:

    @abstractmethod
    def __init__(self, model_path):
        pass

    @abstractmethod
    def predict(self, X) -> float:
        pass
