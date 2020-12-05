import numpy as np
from numba import njit

from .base_model import BaseModel
from util import logistic


class FFMModel(BaseModel):
    def __init__(self, num_latent, num_features, num_fields, reg_lambda, use_linear=True, **kwargs):
        super().__init__(num_features=num_features, reg_lambda=reg_lambda, use_linear=use_linear)
        self.latent_w = np.random.rand(num_fields, num_features, num_latent) * 1 / np.sqrt(num_latent)
        self.grads = np.ones((num_fields, num_features, num_latent))

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

    def _phi(self, x: np.array):
        """
        Sum over bias and linear terms + sum of products of latent vectors
        """
        return _phi(x, self.bias, self.lin_terms.copy(), self.latent_w.copy())

    def predict(self, x):
        # TODO add batch predicting here
        return logistic(self._phi(x))


@njit
def _phi(x,
         bias,
         lin_terms,
         latent_w):
    phi = 0
    if bias and lin_terms:
        phi = 0
        phi += bias
    for feat in [val[1] for val in x]:
        phi += (1 / np.sqrt(2)) * lin_terms[feat]
    for i, (field1, feat1, val1) in enumerate(x):
        for (field2, feat2, val2) in x[i:]:
            if feat1 > len(latent_w[0]) or feat2 > len(latent_w[0]):
                continue  # Skip unknown features
            phi += (1 / 2) * np.dot(latent_w[field2, feat1], latent_w[field1, feat2]) * val1 * val2
    return phi
