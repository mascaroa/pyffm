import time
from typing import Union, Tuple
import numpy as np
from numba import njit
import logging

from .base_engine import BaseEngine

from .model.ffm_model import FFMModel, calc_phi

logger = logging.getLogger(__name__)
logger.setLevel('INFO')


class FFMEngine(BaseEngine):
    def __init__(self, training_params):
        super().__init__(training_params)

    def create_model(self,
                     num_fields,
                     num_features,
                     **training_params):
        self.model = FFMModel(num_fields=num_fields, num_features=num_features, **training_params)

    def train(self,
              x_train: list,
              x_test: Union[list, None] = None) -> int:
        """
        :param x_train: Training data formatted as a list of lists of tuples (rows) like:
                        [[click, (feat1, field1, val1), (feat2, field2, val2), ...],
                        [click, (...), ...]]
                        where click = 0 or 1; featN, fieldN are ints and valN are ints or floats
        :param x_test: Test data formatted the same as the train data
        :return: 0 if trained succesfully
        """
        if self.model is None:
            num_fields = max([val[0] for row in x_train for val in row[1:]]) + 1
            num_features = max([val[1] for row in x_train for val in row[1:]]) + 1
            self.create_model(num_fields=num_fields, num_features=num_features, **self._training_params)

        if not isinstance(x_train, list):
            raise TypeError('x data must be a list data rows!')

        if isinstance(x_train[0], int) or isinstance(x_train[0], tuple):
            x_train = [x_train]

        if type(x_train).__module__ != np.__name__:
            y_train = np.array(np.array([np.array(sub_arr[0]) for sub_arr in x_train]))
            x_train = np.array([np.array(np.array([np.array(val) for val in sub_arr[1:]])) for sub_arr in x_train])

        if x_test is not None:
            if type(x_test).__module__ != np.__name__:
                y_test = np.array(np.array([np.array(sub_arr[0]) for sub_arr in x_test]))
                x_test = np.array([np.array(np.array([np.array(val) for val in sub_arr[1:]])) for sub_arr in x_test])

        full_start = time.time()
        for epoch in range(self.epochs):
            logger.info(f'Epoch {epoch}')
            sample_line = np.random.randint(0, len(x_train) - 1)
            self.model.kappa = (x_train[sample_line], y_train[sample_line])
            logger.info(f'Training on {len(x_train)} rows.')
            start_time = time.time()
            grads, lin_terms, latent_w = full_train(x_train,
                                                    self.model.kappa.copy(),
                                                    self.model.reg_lambda,
                                                    self.learn_rate,
                                                    self.model.grads.copy(),
                                                    self.model.lin_terms.copy(),
                                                    self.model.latent_w.copy())
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

    def predict(self, x):
        return self.model.predict(x)


@njit
def full_train(x_train,
               kappa,
               reg_lambda,
               learn_rate,
               grads,
               lin_terms,
               latent_w) -> Tuple[np.array, np.array, np.array]:
    for x_line in x_train:
        if lin_terms is not None:
            for x_1 in x_line:
                lin_grad = (reg_lambda * lin_terms[x_1[1]] + kappa * x_1[2] * (1 / np.sqrt(2)))
                lin_terms[x_1[1]] = lin_terms[x_1[1]] - learn_rate * lin_grad
        i = 0
        for x_1 in x_line:
            if x_1[2] == 0:
                continue  # Only calculate non-zero valued terms
            for x_2 in x_line[i + 1:]:
                g1 = reg_lambda * latent_w[x_1[0], x_2[1]] + kappa * latent_w[x_2[0], x_1[1]] * x_1[2] * x_2[2]
                g2 = reg_lambda * latent_w[x_2[0], x_1[1]] + kappa * latent_w[x_1[0], x_2[1]] * x_1[2] * x_2[2]
                grads[x_1[0], x_2[1]] += g1 ** 2
                grads[x_2[0], x_1[1]] += g2 ** 2

                latent_w[x_1[0], x_2[1]] -= learn_rate * g1 / np.sqrt(grads[x_1[0], x_2[1]])
                latent_w[x_2[0], x_1[1]] -= learn_rate * g1 / np.sqrt(grads[x_2[0], x_1[1]])
            i += 1
    return grads, lin_terms, latent_w


@njit
def calc_logloss(x_test,
                 y_test,
                 bias,
                 lin_terms,
                 latent_w):
    logloss = 0
    for i, x_line in enumerate(x_test):
        logloss += np.log(1 + np.exp(-y_test[i] * calc_phi(x_line, bias, lin_terms, latent_w)))
    logloss = logloss / len(x_test)
    return logloss
