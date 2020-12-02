from typing import Union
import numpy as np
import logging

from .base_engine import BaseEngine

from .model.fm_model import FMModel

logger = logging.getLogger(__name__)


class FMEngine(BaseEngine):
    def __init__(self, training_params):
        super().__init__(training_params)

    def create_model(self, *args, **kwargs):
        # TODO: figure out params that go in the model vs. in here
        self.model = FMModel(*args, **kwargs)

    def train(self, x_train: list, x_test: Union[list, None] = None) -> int:
        """
        :param x_train: list of lists of tuples (rows) like:
                        [[click, (feat1, val1), (feat2, val2), ...],
                        [click, (...), ...]]
        :param x_test: Test data formatted the same as the train data
        :return: 0 if trained succesfully
        """
        if self.model is None:
            num_features = max([val[1] for row in x_train for val in row[1:]]) + 1
            self.create_model(num_features=num_features, **self._training_params)
        if not isinstance(x_train, list):
            raise TypeError('x data must be a list data rows!')

        if isinstance(x_train[0], int) or isinstance(x_train[0], tuple):
            x_train = [x_train]

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
        return 0

    def predict(self, x):
        pass
