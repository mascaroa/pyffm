import unittest
import numpy as np
import pandas as pd

from pyctr.engine.ctr_engine import CTREngine


class TestCTREngine(unittest.TestCase):
    def setUp(self) -> None:
        self.ctr_engine = CTREngine(model='ffm', training_params={'epochs': 10}, io_params={})

    def test_basic_train(self):
        train_data = [[1, (0, 0, 1), (1, 1, 1), (2, 2, 1), (3, 3, 1)],
                      [1, (0, 4, 1), (1, 5, 1), (2, 2, 1), (3, 3, 1)],
                      [1, (0, 6, 1), (1, 7, 1), (2, 8, 1), (3, 9, 1)]]
        self.ctr_engine.train(train_data)
        print(self.ctr_engine.predict([(0, 0, 1), (1, 1, 1), (2, 2, 1), (3, 3, 1)]))
        print(self.ctr_engine.predict([(0, 4, 1), (1, 5, 1), (2, 2, 1), (3, 3, 1)]))
        print(self.ctr_engine.predict([(0, 6, 1), (1, 7, 1), (2, 8, 1), (3, 9, 1)]))

    def test_big_train(self):
        self.ctr_engine = CTREngine(model='ffm', training_params={'epochs': 10}, io_params={})
        data_in = []
        with open('small_train.txt', 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                click = [int(line.replace('\n', '').split(' ')[0])]
                features = [(int(val.split(':')[0]), int(val.split(':')[1]), float(val.split(':')[2])) for val in line.replace('\n', '').split(' ')[1:]]
                data_in.append(click + features)
        self.ctr_engine.train(data_in)
        self.ctr_engine.predict(data_in[0][1:])
