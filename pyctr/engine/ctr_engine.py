import itertools
from typing import Union
import numpy as np

from pyctr.engine.model.ffm_model import FFMModel
from pyctr.engine.model.fm_model import FMModel
from pyctr.engine.model.poly2_model import Poly2Model

MODEL_DICT = {'ffm': FFMModel,
              'fm': FMModel,
              'poly2': Poly2Model}


class CTREngine:
    def __init__(self, model, training_params, io_params):
        self.model_type = model
        self.model: Union[FFMModel, FMModel, Poly2Model]
        self.epochs = training_params.get('epochs', 10)
        self.learn_rate = training_params.get('learn_rate', 0.001)

    def create_model(self, args, kwargs):
        # TODO: figure out params that go in the model vs. in here
        # Size of model, (num fields, num feats etc.?)
        self.model = MODEL_DICT[self.model_type](*args, *kwargs)

    def train(self, x_data: list):
        if not isinstance(x_data, list):
            raise TypeError('x data must be a list of tuples!')
        for l, x_line in enumerate(x_data):
            y = x_line.pop(0)
            self.model.calc_kappa(x_line, y)
            for i, x_1 in enumerate(x_line):
                for j, x_2 in enumerate(x_line[i:]):
                    g1, g2 = self.model.calc_subgrads(x_1, x_2)
                    self.model.grads[x_1[0], x_2[1]] += g1 ** 2
                    self.model.grads[x_2[0], x_1[1]] += g2 ** 2

                    self.model.latent_w[x_1[0], x_2[1]] -= self.learn_rate * g1 / np.sqrt(self.model.grads[x_1[0], x_2[1]])
                    self.model.latent_w[x_2[0], x_1[1]] -= self.learn_rate * g1 / np.sqrt(self.model.grads[x_2[0], x_1[1]])

    def predict(self, x):
        self.model
