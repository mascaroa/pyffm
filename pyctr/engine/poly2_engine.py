import numpy as np
import logging

from . import BaseEngine

from .model.poly2_model import Poly2Model


class Poly2Engine(BaseEngine):
    def __init__(self, training_params, io_params):
        super().__init__(training_params, io_params)

    def create_model(self, *args, **kwargs):
        # TODO: figure out params that go in the model vs. in here
        self.model = Poly2Model(*args, **kwargs)

    def train(self, x_data: list) -> int:
        """

        :param x_data: Training data formatted as...
        :return:
        """

    def predict(self, x):
        pass
