import numpy as np
import logging

from pyctr.engine.model.ffm_model import FFMModel
from pyctr.engine.model.fm_model import FMModel
from pyctr.engine.model.poly2_model import Poly2Model

MODEL_DICT = {'ffm': FFMModel,
              'fm': FMModel,
              'poly2': Poly2Model}

logger = logging.getLogger(__name__)
logger.setLevel('INFO')


class CTREngine:
    def __init__(self, model, training_params, io_params):
        self.model_type = model
        self.model = None
        self.epochs = training_params.get('epochs', 100)
        self.learn_rate = training_params.get('learn_rate', 0.001)

    def create_model(self, *args, **kwargs):
        # TODO: figure out params that go in the model vs. in here
        # Size of model, (num fields, num feats etc.?)
        self.model = MODEL_DICT[self.model_type](*args, **kwargs)

    def train(self, x_data: list):
        """

        :param x_data: Training data formatted as a list of lists (rows) like:
                        [[click, (feat1, field1, val1), (feat2, field2, val2), ...], [...]]
                        where click = 0 or 1; featN, fieldN are ints and valN are ints or floats
        :return:
        """
        if self.model is None:
            num_features = max([val[0] for row in x_data for val in row[1:]]) + 1
            num_fields = max([val[1] for row in x_data for val in row[1:]]) + 1
            self.create_model(num_latent=4, num_features=num_features, num_fields=num_fields, reg_lambda=0.01)
        if not isinstance(x_data, list):
            raise TypeError('x data must be a list data rows!')
        if isinstance(x_data[0], int) or isinstance(x_data[0], tuple):
            x_data = [x_data]
        for epoch in range(self.epochs):
            logger.info(f'Epoch {epoch}')
            sample_line = np.random.randint(0, len(x_data) - 1)
            self.model.calc_kappa(x_data[sample_line][1:], x_data[sample_line][0])
            for x_line in x_data:
                for i, x_1 in enumerate(x_line[1:]):
                    if x_1[2] == 0:
                        continue  # Only calculate non-zero valued terms
                    for j, x_2 in enumerate(x_line[i + 1:]):
                        g1, g2 = self.model.calc_subgrads(x_1, x_2)
                        self.model.grads[x_1[0], x_2[1]] += g1 ** 2
                        self.model.grads[x_2[0], x_1[1]] += g2 ** 2

                        self.model.latent_w[x_1[0], x_2[1]] -= self.learn_rate * g1 / np.sqrt(self.model.grads[x_1[0], x_2[1]])
                        self.model.latent_w[x_2[0], x_1[1]] -= self.learn_rate * g1 / np.sqrt(self.model.grads[x_2[0], x_1[1]])

    def predict(self, x):
        return self.model.predict(x)
