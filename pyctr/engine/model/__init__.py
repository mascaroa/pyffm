import numpy as np



class BaseModel:
    def __init__(self, num_latent, num_features, num_fields, reg_lambda):
        self.reg_lambda = reg_lambda
        self.bias = 1

        self.lin_terms = np.ones(num_features)
        self.latent_w = np.random.rand([num_features, num_fields, num_latent], 0, 1/np.sqrt(num_latent))

    def kappa(self, x, y):
        return np.divide(-y, (1 + np.exp(y * self._phi(x))))

    def subgrad(self, kappa, j1, f1, j2, f2, x, y):
        """
        :param x:   dict where keys are fields and vals are (feat, val) tuples, e.g.:
                           {0: [(123, 1)],
                            1: [(187, 1), (34, 1)]}
                    only fields with non-zero feature vals must be present
        """
        return self.reg_lambda * self.latent_w[j1, f2] + kappa * self.latent_w[j2, f1] * x.get(j1, [0, 0])[1] * x.get(j2, [0, 0])[1]

    def _phi(self, x):
        raise NotImplementedError('This should only be called from the child classes.')

    def update_weights(self):
        pass

    def compute_logloss(self):
        pass
