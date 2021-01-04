import os
import unittest
import numpy as np
import pandas as pd

from pyffm import PyFFM


class TestCTREngine(unittest.TestCase):
    def test_train_from_file(self):
        training_params = {'epoch': 10, 'reg_lambda': 0.002, 'sigmoid': True, 'parallel': False, 'early_stop': False}
        pyffm = PyFFM(model="ffm", training_params=training_params)

        import os

        dir = os.getcwd()
        pyffm.train(os.path.join(dir, 'data', 'bigdata.tr.txt'))
        preds = pyffm.predict(os.path.join(dir, 'data', 'bigdata.te.txt'))
        test_preds = np.array([0.19198589, 0.36800491, 0.26981428, 0.41630196, 0.2679637,
                               0.33786355, 0.12455032, 0.21487805, 0.28094977, 0.16309123])
        self.assertTrue(np.allclose(test_preds, preds[:10]))

    def test_train_from_datafraome(self):
        training_params = {'epochs': 1, 'reg_lambda': 0.002, 'parallel': True}
        pyffm = PyFFM(model="ffm", training_params=training_params)

        file_path = os.path.join(os.getcwd(), 'data', 'small_sample_train.csv')
        df_in = pd.read_csv(file_path)

        balanced_df = df_in[df_in['click'] == 1].append(df_in[df_in['click'] == 0].sample(n=1000)).sample(frac=1)

        train_data = balanced_df.sample(frac=0.9)
        predict_data = balanced_df.drop(train_data.index)

        pyffm.train(train_data)
        preds = pyffm.predict(predict_data.drop(columns='click'))

        print(preds)

    def test_train_from_list(self):
        list_train_data = [{'click': 1, 'feat1': 'thing', 'feat2': 'thing2', 'feat3': 3.0},
                           {'click': 0, 'feat1': 'thing', 'feat2': 'thing4', 'feat3': 5.7},
                           {'click': 0, 'feat1': 'thing2', 'feat2': 'thing6', 'feat3': 0.1}]

        list_predict_data = [{'feat1': 'thing', 'feat2': 'thing2', 'feat3': 3.0},
                             {'feat1': 'thing', 'feat2': 'thing4', 'feat3': 5.7},
                             {'feat1': 'thing2', 'feat2': 'thing6', 'feat3': 0.1}]

        training_params = {'epochs': 1, 'reg_lambda': 0.002, 'parallel': True}
        pyffm = PyFFM(model="ffm", training_params=training_params)

        pyffm.train(x_train=list_train_data)
        pyffm.predict(list_predict_data)

    def test_save_and_load_model(self):
        training_params = {'epoch': 10, 'reg_lambda': 0.002, 'sigmoid': True, 'parallel': False, 'early_stop': False}
        pyffm = PyFFM(model="ffm", training_params=training_params)

        import os

        dir = os.getcwd()
        pyffm.train(os.path.join(dir, 'data', 'bigdata.tr.txt'))
        preds = pyffm.predict(os.path.join(dir, 'data', 'bigdata.te.txt'))
        pyffm.save_model(overwrite=True)

        pyffm_2 = PyFFM(model='ffm')
        pyffm_2.load_model()
        preds_2 = pyffm.predict(os.path.join(dir, 'data', 'bigdata.te.txt'))
        self.assertTrue(np.allclose(preds, preds_2))
