from typing import Union
import numpy as np
import pandas as pd

from engine import EngineFactory
from engine.base_engine import BaseEngine
from util import Map, run_as_subprocess

import logging

logger = logging.getLogger(__name__)
logger.setLevel('INFO')  # TODO: propagate logging properly from init kwargs


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
              y_train: Union[str, list, pd.DataFrame, None] = None) -> None:
        """

            Train data must be a str (path to libffm file), a list of lists of dictionaries, or a dataframe and must contain
            a lobel column (default 'click'), e.g.:
            [[{'click': 1}, {'field1': 'feature1'}, {'field2', 0.83}, ...]              <- 1st row
            [{'click': 0}, {'field2': 'feature_whatever'}, {'field17': True}, ...]]     <- 2nd row ...

            OR


        """
        assert self._check_inputs(x_train, y_train) == 0
        formatted_x_data, formatted_y_data = self._format_data(x_train, y_train)
        if not self.engine.train_quiet:
            split_frac = self.training_params.get('split_frac', 0.1)
            x_train, y_train, x_test, y_test = self._train_test_split(formatted_x_data, formatted_y_data, split_frac)
            self.engine.num_fields = self.field_map.max() + 1
            self.engine.num_features = self.feature_map.max() + 1
            self.engine.train(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
            return
        self.engine.train(x_train=formatted_x_data, y_train=formatted_y_data)

    def predict(self, x: Union[str, list, pd.DataFrame]) -> np.array:
        assert self._check_inputs(x) == 0
        logger.info('Formatting predict data')
        formatted_predict_data = self._format_data(x, train_or_predict='predict')
        return self.engine.predict(formatted_predict_data)

    def load_model(self, model_path):
        # TODO: Load model from disk
        pass

    def _check_inputs(self, x, y=None) -> int:
        if type(x) in [list, pd.DataFrame, str]:
            if y is not None and type(y) in [list, pd.DataFrame, str]:
                return 0
            return 0
        raise TypeError(f'Input data must be [list, pd.DataFrame, str] not {type(x)}!')

    def _format_data(self,
                     x_data: Union[str, list, pd.DataFrame],
                     y_data: Union[str, list, pd.DataFrame, None] = None,
                     train_or_predict: str = 'train') -> [np.array, np.array]:
        if isinstance(x_data, str):
            return self._format_file_data(x_data, train_or_predict=train_or_predict)
        elif isinstance(x_data, pd.DataFrame):
            return self._format_dataframe(x_data, y_df=y_data, train_or_predict=train_or_predict)
        elif isinstance(x_data, list):
            return self._format_list_data(x_data, y_list=y_data, train_or_predict=train_or_predict)
        raise TypeError(f'Data must be [str, list, pd.DataFrame] not {type(x_data)}')

    @run_as_subprocess
    def _format_dataframe(self,
                          x_df: pd.DataFrame,
                          y_df=None,
                          train_or_predict='train',
                          label_name='click') -> (np.array, np.array):
        # Running as subprocess so memory is freed after (df.copy is *definitely*  temporary this way)
        logger.debug('Formatting dataframe')
        if train_or_predict == 'train':
            field_map_func = self.field_map.add
            feature_map_func = self.feature_map.add
        elif train_or_predict == 'predict':
            field_map_func = self.field_map.get
            feature_map_func = self.feature_map.get
        else:
            raise NameError(f'train_or_predict must be "train" or "predict" not {train_or_predict}!')

        if label_name in x_df.columns:
            assert y_df is None, f'Label column ({label_name}) found in dataframe but y data was also passed!'
            y_data = x_df[label_name].values
            x_train = x_df.drop(columns=label_name)
        else:
            y_data = y_df.copy
            x_train = x_df

        num_cols = len(x_train.columns)
        num_rows = len(x_train)

        arr = np.zeros((num_rows, num_cols, 3))
        fields = x_train.columns.values
        dtypes = x_train.dtypes.values
        x_data = x_train.values
        for i in range(num_rows):
            for j in range(num_cols):
                if 'float' in str(dtypes[j]):
                    arr[i, j, :] = np.array([field_map_func(fields[j]), feature_map_func(x_data[i, j]), x_data[i, j]])
                    continue
                arr[i, j, :] = np.array([field_map_func(fields[j]), feature_map_func(x_data[i, j]), 1])

        if y_data is not None and train_or_predict == 'train':
            return arr, y_data
        return arr

    def _format_list_data(self,
                          x_list: list,
                          y_list: list = None,
                          train_or_predict: str = 'train') -> (np.array, np.array):
        logger.debug('Formatting list data')
        return x_list, y_list

    def _format_file_data(self,
                          filename: str,
                          train_or_predict: str = 'train') -> (np.array, np.array):
        """
        Load preformatted (LibFFM) files for training
        """
        logger.debug('Loading file data')
        if train_or_predict == 'train':
            field_map_func = self.field_map.add
            feature_map_func = self.feature_map.add
        elif train_or_predict == 'predict':
            field_map_func = self.field_map.get
            feature_map_func = self.feature_map.get
        else:
            raise NameError(f'train_or_predict must be "train" or "predict" not {train_or_predict}!')

        x_data, y_data = [], []
        with open(filename, 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                if len(line.split(':')[0].split(' ')) > 1:  # Click values present, parse y data as well
                    y_data.append(np.array([int(line.replace('\n', '').split(' ')[0])]))
                features = []
                for val in line.replace('\n', '').split(' ')[1:]:
                    try:
                        features.append(np.array([int(field_map_func(val.split(':')[0])), int(feature_map_func(val.split(':')[1])), float(val.split(':')[2])]))
                    except TypeError:  # TypeError thrown when field/featuremap get returns None (unknown field/feature)
                        continue
                x_data.append(np.array(features))

        num_cols = max([len(row) for row in x_data])
        num_rows = len(x_data)

        # Zero-pad each row so we can have 1 nice np array
        # This might be super slow?...
        for i in range(len(x_data)):
            if len(x_data[i]) < num_cols:
                x_data[i] = np.vstack((x_data[i], [np.array([0, 0, 0]) for j in range(len(x_data[i]), num_cols)]))
        x_data = np.concatenate(np.concatenate(x_data)).reshape(num_rows, num_cols, 3)
        if y_data is not None and train_or_predict == 'train':
            return x_data, np.concatenate(y_data)
        return x_data

    def _train_test_split(self,
                          x_data,
                          y_data,
                          split_frac) -> (np.array, np.array, np.array, np.array):
        split_index = int(len(x_data) - len(x_data) * split_frac)
        return x_data[:split_index], y_data[:split_index], x_data[split_index:], y_data[split_index:]
