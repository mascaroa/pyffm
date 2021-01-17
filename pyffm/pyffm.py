import os
import pickle
import datetime
from typing import Union
import numpy as np
import pandas as pd

from .engine import EngineFactory
from .engine.base_engine import BaseEngine
from .util import Map

import logging

logger = logging.getLogger(__name__)


class PyFFM:
    """
        Top level class to handle the data formatting, io, etc.
    """

    def __init__(self,
                 model=None,
                 training_params=None,
                 io_params=None,
                 problem='classification',
                 **kwargs):

        self.training_params = {} if training_params is None else training_params
        self.io_params = {} if io_params is None else io_params

        self.model = 'ffm' if model is None else model

        if problem.lower() not in ['classification', 'regression']:
            raise ValueError(f'Problem must be classification or regression not {problem}')
        self.problem = problem.lower()
        if self.problem == 'regression':
            self.training_params['regression'] = True

        if self.model not in EngineFactory:
            raise NameError(f'Model {self.model.lower()} not found! Must be in {EngineFactory}')
        self.engine: BaseEngine
        self.engine = EngineFactory[self.model](training_params=self.training_params)
        self.feature_map = Map()
        self.field_map = Map()

        self.model_dir = self.io_params.get('model_dir', os.path.join(os.getcwd(), 'model'))
        self.model_filename = self.io_params.get('model_filename', 'model.npz')

        self.set_log_level(kwargs.pop('log_level', 'INFO'))

        if len(kwargs):
            logger.warning(f'Unknown keyword args: {kwargs.keys()}')

    def train(self,
              x_train: Union[str, list, pd.DataFrame],
              y_train: Union[str, list, pd.DataFrame, None] = None,
              label_name: str = 'click') -> None:
        """
            Train data must be a str (path to libffm file), a list of dictionaries, or a dataframe and must contain
            a lobel column (default 'click'), e.g.:

            list:

            [{'click': 1, 'field1': 'feature1', 'field2', 0.83, ...},              <- 1st row
            {'click': 0, 'field2': 'feature_whatever', 'field17': True, ...}]     <- 2nd row ...

            DataFrame:

               click    field1    field2  ...  fieldN
                0       163825     2112          3.57
                1       116178     3104          3.14
                ...

            filepath: (str)

            libffm formatted file (see pyffm/test/data/bigdata.tr.txt)
        """
        # TODO: add online learning
        formatted_x_data, formatted_y_data = self._format_data(x_train,
                                                               y_train,
                                                               label_name=label_name)
        if not self.engine.train_quiet:
            split_frac = self.training_params.get('split_frac', 0.1)
            x_train, y_train, x_test, y_test = _train_test_split(formatted_x_data,
                                                                 formatted_y_data,
                                                                 split_frac)
            self.engine.num_fields = self.field_map.max() + 1
            self.engine.num_features = self.feature_map.max() + 1
            self.engine.train(x_train=x_train,
                              y_train=y_train,
                              x_test=x_test,
                              y_test=y_test)
        else:
            self.engine.train(x_train=formatted_x_data,
                              y_train=formatted_y_data)

    def predict(self, x: Union[str, list, pd.DataFrame],
                label_name: str = 'click') -> np.array:
        logger.info('Formatting predict data')
        formatted_predict_data = self._format_data(x, train_or_predict='predict', label_name=label_name)
        return self.engine.predict(formatted_predict_data)

    def load_model(self, model_dir=None):
        if model_dir is None:
            logger.info(f'No model path given, using default: {self.model_dir}')
            model_dir = self.model_dir
        for mapping_name in ['feature_map', 'field_map']:
            map_path = os.path.join(model_dir, mapping_name + '.pkl')
            if not os.path.exists(map_path):
                logger.error(f'No {mapping_name} found at {map_path}!')
                return
            with open(map_path, 'rb') as f:
                setattr(self, mapping_name, pickle.load(f))

        model_path = os.path.join(model_dir, self.model_filename)
        if not os.path.exists(model_path):
            logger.error(f'No model found at {model_path}!')
            return
        self.engine.load_model(model_path)

    def save_model(self, model_dir=None, overwrite=True):
        if model_dir is None:
            logger.info(f'No model path given, using default: {self.model_dir}')
            model_dir = self.model_dir

        if not os.path.exists(model_dir):
            logger.info(f'Creating model directory {model_dir}')
            os.makedirs(model_dir)

        model_path = os.path.join(model_dir, self.model_filename)
        if os.path.exists(model_path) and overwrite is False:  # Save backups like: '.YYYMMDD_mm_ss_model.npz'
            backup_file_path = os.path.join(model_dir, f'.{datetime.datetime.now().strftime("%Y%m%d_%M_%S_")}' + self.model_filename)
            logger.info(f'Backing up model as {backup_file_path}')
            os.rename(model_path, backup_file_path)
        logger.info(f'Saving model to {model_path}')
        self.engine.save_model(model_path)

        for mapping_name in ['feature_map', 'field_map']:
            map_path = os.path.join(model_dir, mapping_name + '.pkl')
            if os.path.exists(map_path) and overwrite is False:  # Save backups like: '.YYYMMDD_mm_ss_feature_map.pkl'
                backup_file_path = os.path.join(model_dir, f'.{datetime.datetime.now().strftime("%Y%m%d_%M_%S_")}' + mapping_name + '.pkl')
                logger.info(f'Backing up {mapping_name} as {backup_file_path}')
                os.rename(map_path, backup_file_path)
            logger.info(f'Saving {mapping_name} as {map_path}')
            with open(map_path, 'wb') as f:
                pickle.dump(getattr(self, mapping_name), f)

    def _format_data(self,
                     x_data: Union[str, list, pd.DataFrame],
                     y_data: Union[str, list, pd.DataFrame, None] = None,
                     train_or_predict: str = 'train',
                     label_name='click') -> [np.array, np.array]:
        if train_or_predict == 'train':
            field_map_func = self.field_map.add
            feature_map_func = self.feature_map.add
        elif train_or_predict == 'predict':
            field_map_func = self.field_map.get
            feature_map_func = self.feature_map.get
        else:
            raise NameError(f'train_or_predict must be "train" or "predict" not {train_or_predict}!')

        if isinstance(x_data, str):
            return _format_file_data(x_data,
                                     train_or_predict=train_or_predict,
                                     field_map_func=field_map_func,
                                     feature_map_func=feature_map_func,
                                     problem=self.problem)
        elif isinstance(x_data, pd.DataFrame):
            return _format_dataframe(x_data,
                                     y_df=y_data,
                                     train_or_predict=train_or_predict,
                                     label_name=label_name,
                                     field_map_func=field_map_func,
                                     feature_map_func=feature_map_func,
                                     problem=self.problem)
        elif isinstance(x_data, list):
            return _format_list_data(x_data,
                                     y_list=y_data,
                                     train_or_predict=train_or_predict,
                                     label_name=label_name,
                                     field_map_func=field_map_func,
                                     feature_map_func=feature_map_func,
                                     problem=self.problem)
        raise TypeError(f'Data must be [str, list, pd.DataFrame] not {type(x_data)}')

    def set_log_level(self, log_level: str):
        if log_level.upper() not in ['DEBUG', 'INFO', 'WARNING', 'ERROR']:
            logger.error(f'Log level must be DEBUG, INFO, WARNING, ERROR not {log_level}')
        logger.setLevel(log_level)
        self.engine.set_log_level(log_level)


def _train_test_split(x_data,
                      y_data,
                      split_frac) -> (np.array, np.array, np.array, np.array):
    split_index = int(len(x_data) - len(x_data) * split_frac)
    return x_data[:split_index], y_data[:split_index], x_data[split_index:], y_data[split_index:]


"""
    Data formatting functions
"""


def _format_dataframe(x_df: pd.DataFrame,
                      y_df=None,
                      train_or_predict='train',
                      label_name='click',
                      field_map_func=None,
                      feature_map_func=None,
                      problem: str = None) -> (np.array, np.array):
    logger.info('Formatting dataframe')
    if label_name in x_df.columns:
        assert y_df is None, f'Label column ({label_name}) found in dataframe but y data was also passed!'
        y_data = x_df[label_name].values
        x_train = x_df.drop(columns=label_name)
    else:
        y_data = y_df.copy if y_df is not None else y_df
        x_train = x_df

    num_cols = len(x_train.columns)
    num_rows = len(x_train)

    x_arr = np.zeros((num_rows, num_cols, 3))
    fields = x_train.columns.values
    dtypes = x_train.dtypes.values
    x_data = x_train.values
    for i in range(num_rows):
        for j in range(num_cols):
            if 'float' in str(dtypes[j]):
                x_arr[i, j, :] = [field_map_func(fields[j]), feature_map_func(x_data[i, j]), x_data[i, j]]
                continue
            x_arr[i, j, :] = [field_map_func(fields[j]), feature_map_func(x_data[i, j]), 1]

    if y_data is not None and train_or_predict == 'train':
        if y_data.min() == 0 and y_data.max() == 1 and problem == 'classification':
            y_data = -1 + y_data * 2
        return x_arr, y_data
    return x_arr


def _format_list_data(x_list: list,
                      y_list: list = None,
                      train_or_predict: str = 'train',
                      label_name='click',
                      field_map_func=None,
                      feature_map_func=None,
                      problem: str = None) -> (np.array, np.array):
    x_list = x_list.copy()
    logger.info('Formatting list data')
    y_data = np.zeros(len(x_list))
    if train_or_predict == 'train' and y_list is None:
        for i, row in enumerate(x_list):
            y_data[i] = row.pop(label_name)

    x_data = np.zeros((len(x_list), max([len(row) for row in x_list]), 3))
    for i, row in enumerate(x_list):
        for j, (key, val) in enumerate(row.items()):
            if isinstance(val, float):
                x_data[i, j, :] = np.array([int(field_map_func(key)), int(feature_map_func(key)), val])
            else:
                x_data[i, j, :] = np.array([int(field_map_func(key)), int(feature_map_func(val)), 1])
    if train_or_predict == 'train':
        if y_data.min() == 0 and y_data.max() == 1 and problem == 'classification':
            y_data = -1 + y_data * 2
        return x_data, y_data
    return x_data


def _format_file_data(filename: str,
                      train_or_predict: str = 'train',
                      field_map_func=None,
                      feature_map_func=None,
                      problem: str = None) -> (np.array, np.array):
    """
        Load preformatted (LibFFM) files for training
    """
    # TODO: add file formatting for FM
    x_data, y_data = [], []
    with open(filename, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            if len(line.split(':')[0].split(' ')) > 1:  # Click values present, parse y data as well
                y_data.append([int(line.replace('\n', '').split(' ')[0])])
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
        y_data = np.concatenate(y_data)
        if y_data.min() == 0 and y_data.max() == 1 and problem == 'classification':
            y_data = -1 + y_data * 2
        return x_data, y_data
    return x_data
