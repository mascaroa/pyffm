import numpy as np


class BaseModel:
    def __init__(self, num_latent, num_features, reg_lambda, linear=False):
        self.reg_lambda = reg_lambda
        self.linear = linear
        self.bias = 1

        self.lin_terms = np.ones(num_features)
        self.latent_w = np.ones(num_latent, num_features)

    def kappa(self, x, y):
        return np.divide(-y, (1 + np.exp(y * self._phi(x))))

    def _phi(self, x):
        raise NotImplementedError('This should only be called from the child classes.')

    def update_weights(self):
        pass

    def compute_logloss(self):
        pass
