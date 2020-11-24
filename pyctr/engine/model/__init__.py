import numpy as np



class BaseModel:
    def __init__(self, num_latent, num_features, num_fields, reg_lambda):
        self.reg_lambda = reg_lambda
        self.bias = 1

        self.lin_terms = np.ones(num_features)
        self.latent_w = np.random.rand([num_features, num_fields, num_latent], 0, 1/np.sqrt(num_latent))
        self.grads = np.ones(num_features, num_fields, num_latent)

        self.square_grad = 0
        self.kappa = 0

    def kappa(self, x, y):
        raise NotImplementedError('This should only be called from the child classes.')

    def subgrad(self, kappa, j1, f1, x1, j2, f2, x2):
        return self.reg_lambda * self.latent_w[j1, f2] + kappa * self.latent_w[j2, f1] * x1 * x2

    def _phi(self, x):
        raise NotImplementedError('This should only be called from the child classes.')

