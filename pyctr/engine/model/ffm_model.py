import itertools as it
from typing import Tuple, List
import numpy as np

from . import BaseModel
from util import logistic


class FFMModel(BaseModel):
    def __init__(self, num_latent, num_features, num_fields, reg_lambda, use_linear=True):
        super().__init__(num_latent=num_latent, num_features=num_features, num_fields=num_fields, reg_lambda=reg_lambda, use_linear=use_linear)

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

    def calc_subgrads(self, x_1, x_2) -> (float, float):
        return self._subgrad(self.kappa, *x_1, *x_2), self._subgrad(self.kappa, *x_2, *x_1)

    def _subgrad(self, kappa, j1, f1, x1, j2, f2, x2):
        return self.reg_lambda * self.latent_w[j1, f2] + kappa * self.latent_w[j2, f1] * x1 * x2

    def calc_lin_subgrads(self, x_1):
        return self._lin_subgrad(self.kappa, x_1[1], x_1[2])

    def _lin_subgrad(self, kappa, f1, x1):
        return self.reg_lambda * self.lin_terms[f1] + kappa * x1 * (1 / np.sqrt(2))

    def _phi(self, x: List[Tuple[int, int, float]]):
        """
        Sum over bias and linear terms + sum of products of latent vectors
        """
        phi = 0
        if self.use_linear:
            phi += self.bias
            for feat in [val[0] for val in x]:
                phi += (1 / np.sqrt(2)) * self.lin_terms[feat]
        for ((field1, feat1, val1), (field2, feat2, val2)) in it.combinations(x, r=2):
            phi += (1 / 2) * np.dot(self.latent_w[field2, feat1], self.latent_w[field1, feat2]) * val1 * val2
        return phi

    def predict(self, x):
        return logistic(self._phi(x))
