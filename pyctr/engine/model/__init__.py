from abc import ABC, abstractmethod

import numpy as np


class BaseModel(ABC):
    def __init__(self, num_latent, num_features, num_fields, reg_lambda, use_linear):
        self.reg_lambda = reg_lambda
        self.bias = 0

        self.use_linear = use_linear
        self.lin_terms = np.zeros(num_features) if use_linear else None
        self.latent_w = np.random.rand(num_fields, num_features, num_latent) * 1 / np.sqrt(num_latent)
        self.grads = np.ones((num_fields, num_features, num_latent))

        self.square_grad = 0
        self.kappa = 0

    @abstractmethod
    def calc_kappa(self, x, y):
        pass

    @abstractmethod
    def _subgrad(self, kappa, j1, f1, x1, j2, f2, x2):
        pass

    @abstractmethod
    def _phi(self, x):
        pass

    @abstractmethod
    def predict(self, x):
        pass
