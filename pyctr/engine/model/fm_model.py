import itertools as it
from typing import Tuple, List
import numpy as np

from .base_model import BaseModel
from util import logistic


class FMModel(BaseModel):
    def __init__(self,
                 num_latent,
                 num_features,
                 reg_lambda,
                 use_linear=False,
                 **kwargs):
        super().__init__(num_features=num_features,
                         reg_lambda=reg_lambda,
                         use_linear=use_linear,
                         sigmoid=kwargs.get('sigmoid', False))
        self.latent_w = np.random.rand(num_features, num_latent) * 1 / np.sqrt(num_latent)
        self.grads = np.ones((num_features, num_latent))

    @property
    def kappa(self):
        return self._kappa

    @kappa.setter
    def kappa(self, value):
        if isinstance(value, int):
            self._kappa = value
        else:
            x, y = value
            self._kappa = np.divide(-y, (1 + np.exp(y * self._phi(x))))

    def calc_subgrad(self, x_1) -> float:
        return self._subgrad(self.kappa, *x_1)

    def _subgrad(self, kappa, f1, x1):
        return self.reg_lambda * self.latent_w[f1] + kappa * x1 * (1 / np.sqrt(2))

    def calc_lin_subgrads(self, x_1):
        return self._lin_subgrad(self.kappa, x_1[0], x_1[1])

    def _lin_subgrad(self, kappa, f1, x1):
        return self.reg_lambda * self.lin_terms[f1] + kappa * x1 * (1 / np.sqrt(2))

    def predict(self, x):
        return logistic(self._phi(x))

    def _phi(self, x: List[Tuple[int, float]]):
        """
        Sum over bias and linear terms + sum of products of latent vectors
        TODO - implement the O(nk) implementation:  Σ((Σ w_j'x_j') - w_j x_j) · w_j x_j
        """
        phi = 0
        if self.use_linear:
            phi += self.bias
            for feat in [val[0] for val in x]:
                phi += (1 / np.sqrt(2)) * self.lin_terms[feat]
        for ((feat1, val1), (feat2, val2)) in it.combinations(x, r=2):
            phi += np.dot(self.latent_w[feat1], self.latent_w[feat2]) * val1 * val2
        return phi

    def logloss(self, x, y):
        return np.log(1 + np.exp(-y * self._phi(x)))
