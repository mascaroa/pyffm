import itertools
from typing import Tuple, List
import numpy as np

from pyctr.engine.model import BaseModel


class FFMModel(BaseModel):
    def __init__(self, num_latent, num_features, num_fields, reg_lambda, linear_terms=False):
        super().__init__(num_latent=num_latent, num_features=num_features, num_fields=num_fields, reg_lambda=reg_lambda)
        self.linear_terms = linear_terms

    def calc_kappa(self, x, y):
        self.kappa = np.divide(-y, (1 + np.exp(y * self._phi(x))))

    def calc_subgrads(self, x_1, x_2) -> (float, float):
        g1 = self.subgrad(self.kappa, *x_1, *x_2)
        g2 = self.subgrad(self.kappa, *x_2, *x_1)
        return g1, g2

    def _phi(self, x: List[Tuple[int, int, float]]):
        """
        Sum over bias and linear terms + sum of products of latent vectors
        """
        phi = 0
        if self.linear_terms:
            phi += self.bias
            for lin_term in self.lin_terms:
                phi += 1 / np.sqrt(2) * lin_term
        combos = itertools.combinations(x, r=2)
        for ((field1, feat1, val1), (field2, feat2, val2)) in combos:
            phi += 1 / np.sqrt(2) * np.dot(self.latent_w[field2, feat1], self.latent_w[field1, feat2]) * val1 * val2
        return phi
