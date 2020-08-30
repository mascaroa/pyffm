import itertools
import numpy as np

from pyctr.engine.model import BaseModel


class FFMModel(BaseModel):
    def __init__(self, num_latent, num_features, reg_lambda, linear=False):
        super().__init__(num_latent=num_latent, num_features=num_features, reg_lambda=reg_lambda, linear=linear)

    def _phi(self, x):
        phi = 0
        if self.linear:
            phi += self.bias
            for lin_term in self.lin_terms:
                phi += 1 / 2 * lin_term
        combos = itertools.combinations(x, r=2)
        for ((field1, feat1, val1), (field2, feat2, val2)) in combos:
            phi += 1 / np.sqrt(2) * np.dot(self.latent_w[field2, feat1], self.latent_w[field1, feat2]) * val1 * val2
        return phi
