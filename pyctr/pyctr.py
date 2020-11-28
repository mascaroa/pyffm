from typing import Union
import pandas as pd

from pyctr.engine.ctr_engine import CTREngine

import logging

logger = logging.getLogger(__name__)


class PyCTR:
    """
        Top level class to handle the data formatting, io, etc.
    """

    def __init__(self, model=None, training_params=None, io_params=None, **kwargs):
        training_params = {} if training_params is None else training_params
        io_params = {} if io_params is None else io_params

        self.train_from_file = io_params.get('train_from_file', False)
        self.predict_from_file = io_params.get('train_from_file', False)
        model = 'FFM' if model is None else model

        self.engine = CTREngine(model, training_params=training_params, io_params=io_params)

    def train(self, data_in: Union[str, list, pd.DataFrame]):
        self._check_inputs(data_in)
        # Format data and train here

    def predict(self, x: Union[str, list, pd.DataFrame]):
        self._check_inputs(x)
        # Format and predict

    def _check_inputs(self, x):
        if type(x) not in [str, list, type(pd.DataFrame)]:
            raise TypeError(f'Predict data must be [str, list, pd.DataFrame] not {type(x)}')
        if isinstance(x, str):
            logger.info('String input detected, training from file')
            self.predict_from_file = True
        elif isinstance(x, pd.DataFrame):
            logger.info('DataFrame input detected')
        elif isinstance(x, str):
            logger.info('List data detected')

    def _format_train_data(self):
        pass
