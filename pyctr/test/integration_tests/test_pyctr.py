import os
import unittest
import numpy as np
import pandas as pd

from pyctr import PyCTR


class TestCTREngine(unittest.TestCase):
    def setUp(self) -> None:
        self.pyctr = PyCTR(model="ffm")

    def test_train_from_file(self):

        import os

        dir = os.getcwd()
        self.pyctr.train(os.path.join(dir, 'small_train.txt'))
        preds = self.pyctr.predict(os.path.join(dir, 'small_train.txt'))
        print(preds)
        # assert some stuff here, write full tests eventually

    def test_train_from_datafraome(self):
        file_path = os.path.join(os.getcwd(), 'sample_df_train.csv')
        df_in = pd.read_csv(file_path, index_col=0)
        # file_path = 'D:\\train_data.csv'
        # for chunk in pd.read_csv(file_path, delimiter='|', index_col=0, iterator=True, chunksize=1_000_000):
        #     df_in = chunk
        #     break
        df_in.reset_index(inplace=True)
        df_in.rename(columns={'label': 'click'}, inplace=True)
        self.pyctr.train(df_in)
        self.pyctr.predict(df_in)
        print(df_in)