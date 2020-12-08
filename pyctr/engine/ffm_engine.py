import time
import math
from typing import Union, Tuple
import numpy as np
from numba import numba as nb, njit
import logging

from .base_engine import BaseEngine

from .model.ffm_model import FFMModel, calc_phi

logger = logging.getLogger(__name__)
logger.setLevel('INFO')


class FFMEngine(BaseEngine):
    def __init__(self, training_params):
        super().__init__(training_params)
        self._num_fields = None
        self._num_features = None

    @property
    def num_fields(self):
        return self._num_fields

    @num_fields.setter
    def num_fields(self, value):
        assert isinstance(value, int), 'Num fields must be an integer!'
        self._num_fields = value

    @property
    def num_features(self):
        return self._num_features

    @num_features.setter
    def num_features(self, value):
        assert isinstance(value, int), 'Num features must be an integer!'
        self._num_features = value

    def create_model(self,
                     num_fields,
                     num_features,
                     **training_params):
        self.model = FFMModel(num_fields=num_fields, num_features=num_features, **training_params)

    def predict(self, x):
        return self.model.predict(x)

    def train(self,
              x_train: np.array,
              y_train: np.array,
              x_test: Union[np.array, None] = None,
              y_test: Union[np.array, None] = None) -> int:
        """

        :param x_train: X training data formatted as an np.array
        :param x_test: X test data formatted the same as the train data - optional
        :param y_train:
        :param y_test:
        :return: 0 if trained succesfully
        """
        if self.model is None:
            if self.num_fields is None:  # If model size not specified, infer from train data
                self.num_fields = max([len(row) for row in x_train])
            if self.num_features is None:
                self.num_features = max([val[0] for row in x_train for val in row[1:]]) + 1

            logger.info(f'Creating ffm model with {self.num_fields} fields and {self.num_features} features.')
            self.create_model(num_fields=self.num_fields, num_features=self.num_features, **self._training_params)

        if type(x_train).__module__ != np.__name__:
            raise TypeError('x data must be an np array!')

        if x_test is not None and type(x_test).__module__ != np.__name__:
            raise TypeError('x data must be an np array!')

        full_start = time.time()
        for epoch in range(self.epochs):
            logger.info(f'Epoch {epoch}')
            sample_line = np.random.randint(0, len(x_train) - 1)
            self.model.kappa = (x_train[sample_line], y_train[sample_line])
            logger.info(f'Training on {len(x_train)} rows.')
            start_time = time.time()
            grads, lin_terms, latent_w = full_train(x_train,
                                                    self.model.kappa,
                                                    self.model.reg_lambda,
                                                    self.learn_rate,
                                                    self.model.grads,
                                                    self.model.lin_terms,
                                                    self.model.latent_w,
                                                    self.model.num_latent)
            logger.info(f'Full train done, took {time.time() - start_time:.1f}s')
            self.model.grads = grads
            self.model.lin_terms = lin_terms
            self.model.latent_w = latent_w
            self.model.bias -= self.model.kappa * self.learn_rate

            # If test data entered, calc logloss
            logger.info('Calculating logloss')
            start_time = time.time()
            if x_test is not None:
                logloss = calc_logloss(x_test,
                                       y_test,
                                       self.model.bias,
                                       self.model.lin_terms.copy(),
                                       self.model.latent_w.copy())
                logger.info(f'Logloss: {logloss}, \nTook {time.time() - start_time:.1f}s')
                # Store this value in the model or engine?

        logger.info(f'Training done, took {time.time() - full_start:.1f}s')
        return 0


@njit(parallel=True, cache=True)
def full_train(x_train,
               kappa,
               reg_lambda,
               learn_rate,
               grads,
               lin_terms,
               latent_w,
               num_latent) -> Tuple[np.array, np.array, np.array]:
    g1 = np.zeros(num_latent)
    g2 = np.zeros(num_latent)
    for i in range(x_train.shape[0]):
        for j_1 in nb.prange(x_train.shape[1]):
            x_1 = x_train[i, j_1]

            if x_1[2] == 0:
                continue

            if lin_terms is not None:
                lin_terms[int(x_1[1])] -= learn_rate * (reg_lambda * lin_terms[int(x_1[1])] + kappa * x_1[2] * (1 / np.sqrt(2)))

            for j_2 in range(j_1 + 1, x_train.shape[1]):
                x_2 = x_train[i, j_2]
                factor = x_1[2] * x_2[2] * kappa
                for k in range(num_latent):  # This is faster than broadcasting for some reason
                    g1[k] = reg_lambda * latent_w[int(x_1[0]), int(x_2[1])][k] + factor * latent_w[int(x_2[0]), int(x_1[1])][k]
                    g2[k] = reg_lambda * latent_w[int(x_2[0]), int(x_1[1])][k] + factor * latent_w[int(x_1[0]), int(x_2[1])][k]
                grads[int(x_1[0]), int(x_2[1])] += g1 * g1
                grads[int(x_2[0]), int(x_1[1])] += g2 * g2

                latent_w[int(x_1[0]), int(x_2[1])] -= learn_rate * g1 / np.sqrt(grads[int(x_1[0]), int(x_2[1])])
                latent_w[int(x_2[0]), int(x_1[1])] -= learn_rate * g2 / np.sqrt(grads[int(x_2[0]), int(x_1[1])])

    return grads, lin_terms, latent_w


@njit(parallel=True, cache=True)
def calc_logloss(x_test,
                 y_test,
                 bias,
                 lin_terms,
                 latent_w):
    logloss = 0
    for i in nb.prange(len(x_test)):
        logloss += np.log(1 + np.exp(-y_test[i] * calc_phi(x_test[i], bias, lin_terms, latent_w)))
    logloss = logloss / len(x_test)
    return logloss
