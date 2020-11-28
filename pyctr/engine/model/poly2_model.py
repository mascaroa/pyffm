from . import BaseModel


class Poly2Model(BaseModel):
    def __init__(self, num_latent, num_features, reg_lambda, use_linear=False):
        super().__init__(num_latent=num_latent, num_features=num_features, reg_lambda=reg_lambda, use_linear=use_linear)

    def calc_kappa(self, x, y):
        pass

    def _subgrad(self, kappa, j1, f1, x1, j2, f2, x2):
        pass

    def _phi(self, x):
        pass

    def predict(self, x):
        pass
