from abc import ABC, abstractmethod


class BaseEngine(ABC):
    def __init__(self, training_params, io_params):
        self.model = None
        self.epochs = training_params.pop('epochs', 10)
        self.learn_rate = training_params.pop('learn_rate', 0.2)
        self._training_params = training_params
        self._io_params = io_params

    @abstractmethod
    def create_model(self, *args, **kwargs):
        pass

    @abstractmethod
    def train(self, x_data):
        pass

    @abstractmethod
    def predict(self, x):
        pass
