from abc import ABC, abstractmethod

import numpy as np


class BaseModel(ABC):
    def __init__(self, num_features, reg_lambda, use_linear):
        self.reg_lambda = reg_lambda

        self.use_linear = use_linear
        self.lin_terms = np.zeros(num_features) if use_linear else None

        self.bias = 0

        self.square_grad = 0
        self.kappa = 0

    @abstractmethod
    def _subgrad(self, *args):
        pass

    @abstractmethod
    def _phi(self, x):
        pass

    @abstractmethod
    def predict(self, x):
        pass
