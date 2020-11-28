from abc import ABC, abstractmethod


class BaseEngine(ABC):
    def __init__(self, training_params, io_params):
        self.model = None
        self.epochs = training_params.get('epochs', 10)
        self.learn_rate = training_params.get('learn_rate', 0.2)

    @abstractmethod
    def create_model(self, *args, **kwargs):
        pass

    @abstractmethod
    def train(self, x_data):
        pass

    @abstractmethod
    def predict(self, x):
        pass
