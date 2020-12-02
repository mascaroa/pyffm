import numpy as np

from . import BaseModel
from util import logistic


class FMModel(BaseModel):
    def __init__(self, num_latent, num_features, reg_lambda, use_linear=False):
        super().__init__(num_features=num_features, reg_lambda=reg_lambda, use_linear=use_linear)


    def _subgrad(self, kappa, f1, x1, f2, x2):
        pass

    def predict(self, x):
        return logistic(self._phi(x))

    def _phi(self, x):
        pass

    def logloss(self, x, y):
        return np.log(1 + np.exp(-y * self._phi(x)))
