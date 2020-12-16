from abc import ABC, abstractmethod

import numpy as np


class BaseModel(ABC):
    def __init__(self,
                 num_features,
                 reg_lambda,
                 use_linear,
                 sigmoid):
        self.sigmoid = sigmoid

        self.reg_lambda = reg_lambda

        self.use_linear = use_linear
        self.lin_terms = np.zeros(num_features) if use_linear else None
        self.lin_grads = np.ones(num_features) if use_linear else None

        self.bias = 0
        self.bias_grad = 1

    @abstractmethod
    def _phi(self, x):
        pass

    @abstractmethod
    def predict(self, x):
        pass
