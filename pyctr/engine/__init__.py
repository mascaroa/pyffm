from abc import ABC, abstractmethod


class BaseEngine(ABC):
    def __init__(self, model, training_params, io_params):
        pass

    @abstractmethod
    def create_model(self, *args, **kwargs):
        pass

    @abstractmethod
    def train(self, x_data):
        pass

    @abstractmethod
    def predict(self, x):
        pass
