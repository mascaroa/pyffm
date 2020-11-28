from typing import Union
import pandas as pd

from engine.ffm_engine import FFMEngine
from engine.fm_engine import FMEngine
from engine.poly2_engine import Poly2Engine

from util import exception_func

import logging

logger = logging.getLogger(__name__)

ENGINE_DICT = {'ffm': FFMEngine,
               'fm': FMEngine,
               'poly2': Poly2Engine}


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

        exception_inputs = {False: {'exception': 'NameError', 'msg': f'Model {model.lower()} not found! Must be in {ENGINE_DICT.keys()}'}}
        self.engine = ENGINE_DICT.get(model.lower(), exception_func)
        self.engine(**exception_inputs.get(model.lower() in ENGINE_DICT, {'training_params': training_params, 'io_params': io_params}))

    def train(self, data_in: Union[str, list, pd.DataFrame]):
        self._check_inputs(data_in)
        formatted_data = self._format_train_data(data_in)
        self.engine.train(formatted_data)

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
        elif isinstance(x, list):
            logger.info('List data detected')

    def _format_train_data(self, data_in: Union[str, list, pd.DataFrame]) -> list:
        if isinstance(data_in, str):
            logger.info('Loading file data')
            return self._format_file_data(data_in)
        elif isinstance(data_in, pd.DataFrame):
            logger.info('Formatting dataframe')
            return self._format_dataframe(data_in)
        elif isinstance(data_in, list):
            logger.info('Formatting list data')
            return self._format_list_data(data_in)

    def _format_dataframe(self, df_in: pd.DataFrame) -> list:
        pass

    def _format_list_data(self, list_in: list) -> list:
        pass

    def _format_file_data(self, filename: str) -> list:
        # TODO: Map features and fields!
        data_in = []
        with open(filename, 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                click = [int(line.replace('\n', '').split(' ')[0])]
                features = [(int(val.split(':')[0]), int(val.split(':')[1]), float(val.split(':')[2])) for val in line.replace('\n', '').split(' ')[1:]]
                data_in.append(click + features)
        return data_in


if __name__ == '__main__':
    pyctr = PyCTR(model="ffm")
    import os

    dir = os.getcwd()
    pyctr.train(os.path.join(dir, 'test/integration_test/small_train.txt'))
