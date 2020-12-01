from typing import Union
import numpy as np
import pandas as pd

from engine.ffm_engine import FFMEngine
from engine.fm_engine import FMEngine
from engine.poly2_engine import Poly2Engine

import logging

logger = logging.getLogger(__name__)
logger.setLevel('INFO')

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

        if model.lower() not in ENGINE_DICT:
            raise NameError(f'Model {model.lower()} not found! Must be in {ENGINE_DICT.keys()}')
        self.engine = ENGINE_DICT[model.lower()](training_params=training_params, io_params=io_params)

    def train(self, data_in: Union[str, list, pd.DataFrame]):
        self._check_inputs(data_in)
        formatted_data = self._format_train_data(data_in)
        self.engine.train(formatted_data)

    def predict(self, x: Union[str, list, pd.DataFrame]):
        self._check_inputs(x)
        formatted_predict_data = self._format_predict_data(x)
        return self.engine.predict(formatted_predict_data)

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
        return df_in

    def _format_list_data(self, list_in: list) -> list:
        return list_in

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

    def _format_predict_data(self, x):
        #  Do something slightly different here?
        if isinstance(data_in, str):
            logger.info('Loading file data')
            return self._format_file_data(x)
        elif isinstance(data_in, pd.DataFrame):
            logger.info('Formatting dataframe')
            return self._format_dataframe(x)
        elif isinstance(data_in, list):
            logger.info('Formatting list data')
            return self._format_list_data(x)


if __name__ == '__main__':
    pyctr = PyCTR(model="ffm")
    import os

    dir = os.getcwd()
    pyctr.train(os.path.join(dir, 'test/integration_tests/small_train.txt'))
    data_in = []
    with open(os.path.join(dir, 'test/integration_tests/small_train.txt'), 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            click = [int(line.replace('\n', '').split(' ')[0])]
            features = [(int(val.split(':')[0]), int(val.split(':')[1]), float(val.split(':')[2])) for val in line.replace('\n', '').split(' ')[1:]]
            data_in.append(click + features)
    c_m = [[0, 0], [0, 0]]
    all_pos = 0
    for row in data_in:
        pred = pyctr.predict(row[1:])
        if row[0] == 1 and row[0] == np.round(pred):
            c_m[0][0] += 1
        elif row[0] == 0 and row[0] == np.round(pred):
            c_m[1][1] += 1
        elif row[0] == 1 and row[0] != np.round(pred):
            c_m[1][0] += 1
        elif row[0] == 0 and row[0] != np.round(pred):
            c_m[0][1] += 1
        if row[0] == 1:
            all_pos += 1
        print(f'{row[0]} - {int(np.round(pred))}')
    print(f' {c_m[0][0]} | {c_m[0][1]} \n---------- \n  {c_m[1][0]} | {c_m[1][1]}')
