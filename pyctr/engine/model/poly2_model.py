from pyctr.engine.model import BaseModel


class Poly2Model(BaseModel):
    def __init__(self, num_latent, num_features, reg_lambda, linear=False):
        super().__init__(num_latent=num_latent, num_features=num_features, reg_lambda=reg_lambda, linear=linear)

    def _phi(self, x):
        pass
