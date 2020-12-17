from typing import Union
import numpy as np
import logging

from .base_engine import BaseEngine

from .model.fm_model import FMModel

logger = logging.getLogger(__name__)
logger.setLevel('INFO')


# TODO: rewrite using numba
class FMEngine(BaseEngine):
    def __init__(self, training_params):
        logger.info(f'Initializing FM Engine with inputs {training_params}')
        super().__init__(training_params)

    def create_model(self,
                     num_features,
                     **training_params):
        self.model = FMModel(num_features=num_features, **training_params)

    def train(self,
              x_train: np.array,
              y_train: Union[np.array, None] = None,
              x_test: Union[np.array, None] = None,
              y_test: Union[np.array, None] = None) -> int:
        if self.model is None:
            num_features = max([val[1] for row in x_train for val in row[1:]]) + 1
            self.create_model(num_features=num_features, **self._training_params)

        if type(x_train).__module__ != np.__name__:
            raise TypeError('x data must be an np array!')

        if x_test is not None and type(x_test).__module__ != np.__name__:
            raise TypeError('x data must be an np array!')

        for epoch in self.epochs:
            logger.info(f'Epoch {epoch}')
            sample_line = np.random.randint(0, len(x_train) - 1)
            self.model.kappa = (x_train[sample_line][1:], x_train[sample_line][0])
            for x_line in x_train:
                assert x_line[0] in [0, 1], f'Click must be 0 or 1, not {x_line[0]}!'
                if self.model.use_linear:
                    for x_1 in x_line[1:]:
                        gl = self.model.calc_lin_subgrads(x_1)
                        self.model.lin_terms[x_1[1]] -= self.learn_rate * gl
                for i, x_1 in enumerate(x_line[1:]):
                    assert len(x_1) == 2, f'x must be a tuple like (field, val) not {x_1}!'
                    if x_1[1] == 0:
                        continue  # Only calculate non-zero valued terms
                    g = self.model.calc_subgrads(x_1)
                    self.model.grads[x_1[0]] += g ** 2
                    self.model.latent_w[x_1[0]] -= self.learn_rate * g / np.sqrt(self.model.grads[x_1[0]])

            self.model.bias -= self.model.kappa * self.learn_rate

        return 0

    def predict(self, x):
        pass
