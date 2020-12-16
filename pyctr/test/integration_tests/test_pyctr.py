import os
import unittest
import numpy as np
import pandas as pd

from pyctr import PyCTR


class TestCTREngine(unittest.TestCase):
    def setUp(self) -> None:
        training_params = {'epoch': 10, 'reg_lambda': 0.002, 'sigmoid': True, 'parallel': False, 'early_stop': False}
        self.pyctr = PyCTR(model="ffm", training_params=training_params)

    def test_train_from_file(self):
        import os

        dir = os.getcwd()
        self.pyctr.train(os.path.join(dir, 'bigdata.tr.txt'))
        preds = self.pyctr.predict(os.path.join(dir, 'bigdata.te.txt'))
        test_preds = np.array([0.19198589, 0.36800491, 0.26981428, 0.41630196, 0.2679637,
                               0.33786355, 0.12455032, 0.21487805, 0.28094977, 0.16309123])
        self.assertTrue(np.allclose(test_preds, preds[:10]))

    def test_train_from_datafraome(self):
        training_params = {'epoch': 10, 'reg_lambda': 0.002, 'sigmoid': True, 'parallel': True}
        self.pyctr = PyCTR(model="ffm", training_params=training_params)

        file_path = os.path.join(os.getcwd(), 'sample_df_train.csv')
        df_in = pd.read_csv(file_path, index_col=0)
        df_in.rename(columns={'label': 'click'}, inplace=True)

        self.pyctr.train(df_in)
        preds = self.pyctr.predict(df_in)

        print(preds)
