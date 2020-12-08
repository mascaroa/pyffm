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

    def __init__(self,
                 model=None,
                 training_params=None,
                 io_params=None,
                 **kwargs):
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

    def train(self,
              x_train: Union[str, list, pd.DataFrame],
              y_train: Union[str, list, pd.DataFrame, None] = None):
        """

        :param x_train:
        :param y_train:
        :return:
        """
        assert self._check_inputs(x_train, y_train) == 0
        formatted_x_data, formatted_y_data = self._format_train_data(x_train, y_train)
        if not self.engine.train_quiet:
            split_frac = self.training_params.get('split_frac', 0.1)
            x_train, y_train, x_test, y_test = self._train_test_split(formatted_x_data, formatted_y_data, split_frac)
            self.engine.num_fields = self.field_map.max() + 1
            self.engine.num_features = self.feature_map.max() + 1
            self.engine.train(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
            return 0
        self.engine.train(x_train=formatted_x_data, y_train=formatted_y_data)

    def predict(self, x: Union[str, list, pd.DataFrame]):
        """

        :param x:
        :return:
        """
        assert self._check_inputs(x) == 0
        formatted_predict_data = self._format_predict_data(x)
        return self.engine.predict(formatted_predict_data)

        # Format and predict

    def _check_inputs(self, x, y=None):
        if type(x) in [list, pd.DataFrame, str]:
            if y is not None and type(y) in [list, pd.DataFrame, str]:
                return 0
            return 0
        raise TypeError(f'Input data must be [list, pd.DataFrame, str] not {type(x)}!')

    def _format_train_data(self,
                           x_data: Union[str, list, pd.DataFrame],
                           y_data: Union[str, list, pd.DataFrame, None] = None) -> [np.array, np.array]:
        if type(x_data) not in [str, list, pd.DataFrame]:
            raise TypeError(f'Data must be [str, list, pd.DataFrame] not {type(x_data)}')
        if isinstance(x_data, str):
            return self._format_file_data(x_data)
        elif isinstance(x_data, pd.DataFrame):
            return self._format_dataframe(x_data, y_data)
        elif isinstance(x_data, list):
            return self._format_list_data(x_data, y_data)

    def _format_dataframe(self,
                          x_df: pd.DataFrame,
                          y_df=None,
                          label_name='click') -> (np.array, np.array):
        """
        :param x_df: X data (dataframe)
        :param y_df: Y data (dataframe) - optional if y data is in X data already
        :param label_name: Name of label column, not used if Y data inputted separately
        :return:
        """
        logger.debug('Formatting dataframe')
        for col in [col for col in x_df.columns if col != label_name]:
            if 'float' not in str(x_df[col].dtype):
                x_df[col] = x_df[col].apply(lambda x: np.array([self.field_map.add(col), self.feature_map.add(x), 1 if not pd.isna(x) else 0]))
            else:
                x_df[col] = x_df[col].apply(lambda x: np.array([self.field_map.add(col), self.feature_map.add(x), x]))

        if y_df is None:
            assert label_name in x_df.columns, f'Label column ({label_name}) must be in dataframe if y data is not passed separately!'
            y_data = x_df[label_name].values
            x_df.drop(columns=label_name, inplace=True)
        else:
            y_data = y_df

        # x_df.rename(columns={col: self.field_map.get(col) for col in x_df.columns}, inplace=True)
        x_data = x_df.to_numpy()
        num_cols = len(x_data[0])
        num_rows = len(x_data)
        x_data = np.concatenate(np.concatenate(x_data)).reshape(num_rows, num_cols, 3)
        return x_data, y_data

    def _format_list_data(self,
                          x_list_in: list,
                          y_list_in: list = None) -> (np.array, np.array):
        """
        :param x_list_in: List of x data like: [(field, feature, value), (...)]
        :param y_list_in: List of y data - optional if y data
        :return:
        """
        logger.debug('Formatting list data')
        return x_list_in, y_list_in

    def _format_file_data(self, filename: str) -> (np.array, np.array):
        """
        Load preformatted (LibFFM) files for training
        """
        logger.debug('Loading file data')
        # TODO: Map features and fields!
        x_data = []
        y_data = []
        with open(filename, 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                if len(line.split(':')[0].split(' ')) > 1:  # Click values present, parse y data as well
                    y_data.append(np.array([int(line.replace('\n', '').split(' ')[0])]))
                features = np.array([np.array([int(self.field_map.add(val.split(':')[0])), int(self.feature_map.add(val.split(':')[1])), float(val.split(':')[2])]) for val in line.replace('\n', '').split(' ')[1:]])
                x_data.append(features)

        num_cols = max([len(row) for row in x_data])
        num_rows = len(x_data)

        # Zero-pad each row so we can have 1 nice np array
        # This is probably super slow... TODO: fix it
        for i in range(len(x_data)):
            if len(x_data[i]) < num_cols:
                x_data[i] = np.vstack((x_data[i], [np.array([0, 0, 0]) for j in range(len(x_data[i]), num_cols)]))
        x_data = np.concatenate(np.concatenate(x_data)).reshape(num_rows, num_cols, 3)
        y_data = np.concatenate(y_data) if len(y_data) else None
        return x_data, y_data

    def _train_test_split(self,
                          x,
                          y,
                          split_frac) -> (np.array, np.array, np.array, np.array):
        """

        :param data_in:
        :param split_frac:
        :return: x_train, y_train, x_test, y_test
        """
        split_index = int(len(x) - len(x) * split_frac)
        return x[:split_index], y[:split_index], x[split_index:], y[split_index:]

    def _format_predict_data(self, x):
        """

        :param x:
        :return:
        """
        #  Do something slightly different here?
        if isinstance(x, str):
            logger.debug('Loading file data')
            return self._format_file_data(x)[0]  # Only return x data for predicting
        elif isinstance(x, pd.DataFrame):
            logger.debug('Formatting dataframe')
            return self._format_dataframe(x)
        elif isinstance(x, list):
            logger.debug('Formatting list data')
            return self._format_list_data(x)
