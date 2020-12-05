from typing import Union
import numpy as np
import logging

from .base_engine import BaseEngine

from .model.ffm_model import FFMModel

logger = logging.getLogger(__name__)
logger.setLevel('INFO')


class FFMEngine(BaseEngine):
    def __init__(self, training_params):
        super().__init__(training_params)

    def create_model(self, num_fields, num_features, **training_params):
        self.model = FFMModel(num_fields=num_fields, num_features=num_features, **training_params)

    def train(self, x_train: list, x_test: Union[list, None] = None) -> int:
        """
        :param x_train: Training data formatted as a list of lists of tuples (rows) like:
                        [[click, (feat1, field1, val1), (feat2, field2, val2), ...],
                        [click, (...), ...]]
                        where click = 0 or 1; featN, fieldN are ints and valN are ints or floats
        :param x_test: Test data formatted the same as the train data
        :return: 0 if trained succesfully
        """
        if self.model is None:
            num_fields = max([val[0] for row in x_train for val in row[1:]]) + 1
            num_features = max([val[1] for row in x_train for val in row[1:]]) + 1
            self.create_model(num_fields=num_fields, num_features=num_features, **self._training_params)

        if not isinstance(x_train, list):
            raise TypeError('x data must be a list data rows!')

        if isinstance(x_train[0], int) or isinstance(x_train[0], tuple):
            x_train = [x_train]

        for epoch in range(self.epochs):
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

            # If test data entered, calc logloss
            if x_test:
                logloss = 0
                for x_line in x_test:
                    assert x_line[0] in [0, 1], f'Click must be 0 or 1, not {x_line[0]}!'
                    logloss += self.model.logloss(x_line[1:], x_line[0])
                logloss = logloss / len(x_test)
                logger.info(f'Logloss: {logloss}')

        return 0

    def predict(self, x):
        return self.model.predict(x)
