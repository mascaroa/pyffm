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
        if len(x.shape) > 2:
            logger.info('Batch predicting...')
            preds = np.zeros(x.shape[0])
            for i in range(x.shape[0]):
                preds[i] = self.model.predict(x[i])
            return preds
        return self.model.predict(x)

    def train(self,
              x_train: np.array,
              y_train: Union[np.array, None] = None,
              x_test: Union[np.array, None] = None,
              y_test: Union[np.array, None] = None) -> int:
        if self.model is None:
            if self.num_fields is None:  # If model size not specified, infer from train data
                self.num_fields = int(max([val[0] for row in x_train for val in row[1:]])) + 1
            if self.num_features is None:
                self.num_features = int(max([val[1] for row in x_train for val in row[1:]])) + 1

            logger.info(f'Creating ffm model with {self.num_fields} fields and {self.num_features} features.')
            self.create_model(num_fields=self.num_fields,
                              num_features=self.num_features,
                              **self._training_params)

        if type(x_train).__module__ != np.__name__:
            raise TypeError('x data must be an np array!')

        if x_test is not None and type(x_test).__module__ != np.__name__:
            raise TypeError('x data must be an np array!')

        full_start = time.time()
        for epoch in range(self.epochs):
            logger.info(f'Epoch {epoch}')
            logger.info(f'Training on {len(x_train)} rows.')
            start_time = time.time()
            norms = 1 / (x_train * x_train)[:, :, 2].sum(axis=1)
            globals()['parallel'] = self.parallel  # Need to use globals so numba can cache the jitted func
            globals()['prange_func'] = nb.prange if parallel else range
            self.model.bias, self.model.bias_grad = run_epoch(x_train,
                                                              y_train,
                                                              self.model.latent_w,
                                                              self.model.grads,
                                                              self.model.lin_terms,
                                                              self.model.lin_grads,
                                                              self.model.bias,
                                                              self.model.bias_grad,
                                                              self.model.num_latent,
                                                              self.model.reg_lambda,
                                                              self.learn_rate,
                                                              norms)
            logger.info(f'Full train done, took {time.time() - start_time:.1f}s')

            # If test data entered, calc logloss
            logger.info('Calculating logloss')
            start_time = time.time()
            if x_test is not None:
                logloss = calc_logloss(x_test,
                                       y_test,
                                       self.model.bias,
                                       self.model.lin_terms,
                                       self.model.latent_w)
                logger.info(f'Logloss: {logloss}, \nTook {time.time() - start_time:.1f}s')
                if self.best_loss is None or logloss < self.best_loss:
                    self.best_loss = logloss
                    self.best_loss_epoch = epoch
                elif logloss > self.best_loss and self.early_stop:
                    logger.info(f'Increasing loss detected, early stopping')  # TODO: do the rerun thing here
                    break

        logger.info(f'Training done, took {time.time() - full_start:.1f}s')
        return 0


def run_epoch(*args, **kwargs):
    """
        Output is non-deterministic if done in parallel,
        so provide it as an input option (much slower w/o parallel)
    """
    global parallel
    global prange_func

    @njit(parallel=parallel, cache=True)
    def full_train(x_train,
                   y_train,
                   latent_w,
                   w_grads,
                   lin_terms,
                   lin_grads,
                   bias,
                   bias_grad,
                   num_latent,
                   reg_lambda,
                   learn_rate,
                   norms) -> [int, int]:
        """
            Run one full training epoch while updating model params
        """
        g1 = np.zeros(num_latent)
        g2 = np.zeros(num_latent)
        for i in np.arange(x_train.shape[0]):
            kappa = np.divide(-y_train[i], (1 + np.exp(y_train[i] * calc_phi(x_train[i], bias, lin_terms, latent_w, norms[i]))))
            for j_1 in prange_func(x_train.shape[1]):
                field1, feat1, val1 = x_train[i, j_1]

                if val1 == 0:
                    continue

                if lin_terms is not None:
                    gl = reg_lambda * lin_terms[int(feat1)] + kappa * val1 * np.sqrt(norms[i])
                    lin_grads[int(feat1)] += gl * gl
                    lin_terms[int(feat1)] -= learn_rate * gl / np.sqrt(lin_grads[int(feat1)])

                for j_2 in range(j_1 + 1, x_train.shape[1]):
                    field2, feat2, val2 = x_train[i, j_2]

                    factor = val1 * val2 * kappa * norms[i]
                    for k in range(num_latent):  # This is faster than broadcasting for some reason
                        g1[k] = reg_lambda * latent_w[int(field1), int(feat2)][k] + factor * latent_w[int(field2), int(feat1)][k]
                        g2[k] = reg_lambda * latent_w[int(field2), int(feat1)][k] + factor * latent_w[int(field1), int(feat2)][k]
                    w_grads[int(field1), int(feat2)] += g1 * g1
                    w_grads[int(field2), int(feat1)] += g2 * g2

                    latent_w[int(field1), int(feat2)] -= learn_rate * g1 / np.sqrt(w_grads[int(field1), int(feat2)])
                    latent_w[int(field2), int(feat1)] -= learn_rate * g2 / np.sqrt(w_grads[int(field2), int(feat1)])
            bias_grad += kappa * kappa
            bias -= learn_rate * kappa / np.sqrt(bias_grad)
        return bias, bias_grad

    return full_train(*args, **kwargs)


@njit(parallel=True, cache=True)
def calc_logloss(x_test,
                 y_test,
                 bias,
                 lin_terms,
                 latent_w):
    logloss = 0
    for i in nb.prange(len(x_test)):
        norm = 1 / x_test[i].sum(axis=0)[2]
        logloss += np.log(1 + np.exp(-y_test[i] * calc_phi(x_test[i], bias, lin_terms, latent_w, norm)))
    logloss = logloss / len(x_test)
    return logloss
