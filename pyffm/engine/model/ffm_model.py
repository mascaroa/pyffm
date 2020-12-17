import numpy as np
from numba import njit, numba as nb

from .base_model import BaseModel
from util import logistic


class FFMModel(BaseModel):
    def __init__(self,
                 num_latent,
                 num_features,
                 num_fields,
                 reg_lambda,
                 use_linear=True,
                 **kwargs):
        super().__init__(num_features=num_features,
                         reg_lambda=reg_lambda,
                         use_linear=use_linear,
                         sigmoid=kwargs.get('sigmoid', False))
        self.num_latent = num_latent
        self.grads = np.ones((num_fields, num_features, num_latent))
        np.random.seed(42)  # Not thread safe, but it's only used here
        self.latent_w = np.random.rand(num_fields, num_features, num_latent) * 1 / np.sqrt(num_latent)

    def predict(self, x):
        if self.sigmoid:
            return logistic(self._phi(x))
        return 1 if self._phi(x) > 0 else 0

    def _phi(self, x: np.array):
        return calc_phi(x, self.bias, self.lin_terms, self.latent_w, 1 / (x * x).sum(axis=0)[2])


@njit
def calc_phi(x,
             bias,
             lin_terms,
             latent_w,
             norm):
    """
        Sum over bias and linear terms + sum of products of latent vectors
    """
    phi = 0
    if bias is not None:
        phi += bias
    for i in range(x.shape[0]):
        field1, feat1, val1 = x[i]
        if val1 == 0:
            continue
        if lin_terms is not None:
            phi += np.sqrt(norm) * lin_terms[int(feat1)] * val1
        for j in range(i + 1, x.shape[0]):
            field2, feat2, val2 = x[j]
            if val2 == 0:
                continue
            if feat1 > latent_w.shape[1] or feat2 > latent_w.shape[1]:
                continue
            factor = val1 * val2 * norm
            for k in range(latent_w[int(field2), int(feat1)].size):
                phi += latent_w[int(field2), int(feat1)][k] * latent_w[int(field1), int(feat2)][k] * factor
    return phi
