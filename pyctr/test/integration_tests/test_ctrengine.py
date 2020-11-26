import unittest
import numpy as np

from pyctr.engine.ctr_engine import CTREngine


class TestCTREngine(unittest.TestCase):
    def setUp(self) -> None:
        self.ctr_engine = CTREngine(model='ffm', training_params={}, io_params={})

    def test_basic_train(self):
        train_data = [[1, (0, 0, 1), (1, 1, 1), (2, 2, 1), (3, 3, 1)],
                      [1, (0, 4, 1), (1, 5, 1), (2, 2, 1), (3, 3, 1)]]
        self.ctr_engine.train(train_data)
        print(self.ctr_engine.predict([(0, 0, 1), (1, 1, 1), (2, 2, 1), (3, 3, 1)]))
        print(self.ctr_engine.predict([(0, 4, 1), (1, 5, 1), (2, 2, 1), (3, 3, 1)]))
        print(True)