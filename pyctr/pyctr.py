from typing import Union
import numpy as np
import pandas as pd

from engine import EngineFactory
from engine.base_engine import BaseEngine
from util import Map

import logging

logger = logging.getLogger(__name__)
logger.setLevel('INFO')


class PyCTR:
    """
        Top level class to handle the data formatting, io, etc.
    """

    def __init__(self, model=None, training_params=None, io_params=None, **kwargs):
        self.training_params = {} if training_params is None else training_params
        self.io_params = {} if io_params is None else io_params

        self.train_from_file = self.io_params.get('train_from_file', False)
        self.predict_from_file = self.io_params.get('train_from_file', False)
        self.model = 'ffm' if model is None else model

        if self.model not in EngineFactory:
            raise NameError(f'Model {self.model.lower()} not found! Must be in {EngineFactory}')
        self.engine: BaseEngine
        self.engine = EngineFactory[self.model](training_params=self.training_params)
        self.feature_map = Map()
        self.field_map = Map()

    def train(self, data_in: Union[str, list, pd.DataFrame]):
        """

        :param data_in:
        :return:
        """
        self._check_inputs(data_in)
        formatted_data = self._format_train_data(data_in)
        if not self.engine.train_quiet:
            split_frac = self.training_params.get('split_frac', 0.1)
            test_data, train_data = self._train_test_split(formatted_data, split_frac)
            self.engine.train(x_train=train_data, x_test=test_data)
            return 0
        self.engine.train(x_train=formatted_data)

    def predict(self, x: Union[str, list, pd.DataFrame]):
        """

        :param x:
        :return:
        """
        self._check_inputs(x)
        formatted_predict_data = self._format_predict_data(x)
        return self.engine.predict(formatted_predict_data)

        # Format and predict

    def _check_inputs(self, x):
        if type(x) not in [str, list, pd.DataFrame]:
            raise TypeError(f'Predict data must be [str, list, pd.DataFrame] not {type(x)}')
        if isinstance(x, str):
            logger.debug('String input detected, training from file')
            self.predict_from_file = True
        elif isinstance(x, pd.DataFrame):
            logger.debug('DataFrame input detected')
        elif isinstance(x, list):
            logger.debug('List data detected')

    def _format_train_data(self, data_in: Union[str, list, pd.DataFrame]) -> list:
        """
        
        :param data_in:
        :return:
        """
        if isinstance(data_in, str):
            logger.debug('Loading file data')
            return self._format_file_data(data_in)
        elif isinstance(data_in, pd.DataFrame):
            logger.debug('Formatting dataframe')
            return self._format_dataframe(data_in)
        elif isinstance(data_in, list):
            logger.debug('Formatting list data')
            return self._format_list_data(data_in)

    def _format_dataframe(self, df_in: pd.DataFrame) -> list:
        """

        :param df_in:
        :return:
        """
        for col in [col for col in df_in.columns if col != 'click']:
            if 'float' not in str(df_in[col].dtype):
                df_in[col] = df_in[col].apply(lambda x: (self.feature_map.add(x), 1))
            else:
                df_in[col] = df_in[col].apply(lambda x: (self.feature_map.add(x), x))

        df_in.rename(columns={col: self.field_map.add(col) for col in df_in.columns}, inplace=True)
        data_dict = list(df_in.T.to_dict().values())
        data_list = [tuple(val.items()) for val in data_dict]
        data_list = [[vals[0][1]] + [(val[0], *val[1]) for val in vals[1:]] for vals in data_list]
        return data_list

    def _format_list_data(self, list_in: list) -> list:
        """

        :param list_in:
        :return:
        """
        return list_in

    def _format_file_data(self, filename: str) -> list:
        """

        :param filename:
        :return:
        """
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

    def _train_test_split(self, data_in, split_frac) -> (list, list):
        """

        :param data_in:
        :param split_frac:
        :return:
        """
        split_index = int(len(data_in) - len(data_in) * split_frac)
        return data_in[:split_index], data_in[split_index:]

    def _format_predict_data(self, x):
        """

        :param x:
        :return:
        """
        #  Do something slightly different here?
        if isinstance(x, str):
            logger.debug('Loading file data')
            return self._format_file_data(x)
        elif isinstance(x, pd.DataFrame):
            logger.debug('Formatting dataframe')
            return self._format_dataframe(x)
        elif isinstance(x, list):
            logger.debug('Formatting list data')
            return self._format_list_data(x)
