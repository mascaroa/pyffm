import numpy as np
import logging

from . import BaseEngine

from .model.fm_model import FMModel


class FMEngine(BaseEngine):
    def __init__(self, training_params, io_params):
        super().__init__(training_params, io_params)

    def create_model(self, *args, **kwargs):
        # TODO: figure out params that go in the model vs. in here
        self.model = FMModel(*args, **kwargs)

    def train(self, x_data: list) -> int:
        """

        :param x_data: Training data formatted as...
        :return:
        """

    def predict(self, x):
        pass
