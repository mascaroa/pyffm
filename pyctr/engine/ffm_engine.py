import numpy as np
import logging

from . import BaseEngine

from .model.ffm_model import FFMModel

logger = logging.getLogger(__name__)


class FFMEngine(BaseEngine):
    def __init__(self, training_params, io_params):
        super().__init__(training_params, io_params)

    def create_model(self, *args, **kwargs):
        # TODO: figure out params that go in the model vs. in here
        # Size of model, (num fields, num feats etc.?)
        self.model = FFMModel(*args, **kwargs)

    def train(self, x_data: list) -> int:
        """

        :param x_data: Training data formatted as a list of lists of tuples (rows) like:
                        [[click, (feat1, field1, val1), (feat2, field2, val2), ...],
                        [click, (...), ...]]
                        where click = 0 or 1; featN, fieldN are ints and valN are ints or floats
        :return:
        """
        if self.model is None:
            num_fields = max([val[0] for row in x_data for val in row[1:]]) + 1
            num_features = max([val[1] for row in x_data for val in row[1:]]) + 1
            self.create_model(num_latent=8, num_features=num_features, num_fields=num_fields, reg_lambda=0.01)
        if not isinstance(x_data, list):
            raise TypeError('x data must be a list data rows!')
        if isinstance(x_data[0], int) or isinstance(x_data[0], tuple):
            x_data = [x_data]
        for epoch in range(self.epochs):
            logger.info(f'Epoch {epoch}')
            sample_line = np.random.randint(0, len(x_data) - 1)
            self.model.calc_kappa(x_data[sample_line][1:], x_data[sample_line][0])
            for x_line in x_data:
                assert x_line[0] in [0, 1], f'Click must be 0 or 1, not {x_line[0]}!'
                if self.model.use_linear:
                    for x_1 in x_line[1:]:
                        gl = self.model.calc_lin_subgrads(x_1)
                        self.model.lin_terms[x_1[1]] -= self.learn_rate * gl
                for i, x_1 in enumerate(x_line[1:]):
                    assert len(x_1) == 3, f'x must be a tuple like (field, feat, val) not {x_1}!'
                    if x_1[2] == 0:
                        continue  # Only calculate non-zero valued terms
                    for j, x_2 in enumerate(x_line[i + 1:]):
                        g1, g2 = self.model.calc_subgrads(x_1, x_2)
                        self.model.grads[x_1[0], x_2[1]] += g1 ** 2
                        self.model.grads[x_2[0], x_1[1]] += g2 ** 2

                        self.model.latent_w[x_1[0], x_2[1]] -= self.learn_rate * g1 / np.sqrt(self.model.grads[x_1[0], x_2[1]])
                        self.model.latent_w[x_2[0], x_1[1]] -= self.learn_rate * g1 / np.sqrt(self.model.grads[x_2[0], x_1[1]])
            self.model.bias -= self.model.kappa * self.learn_rate
        return 0

    def predict(self, x):
        return self.model.predict(x)

