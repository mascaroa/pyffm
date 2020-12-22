import unittest
import numpy as np

from pyffm.engine.ffm_engine import FFMEngine


class TestEngine(unittest.TestCase):
    def setUp(self) -> None:
        self.ctr_engine = FFMEngine(training_params={'epochs': 10})

    def test_basic_train(self):
        x_data = np.array([[(0, 0, 1), (1, 1, 1), (2, 2, 1), (3, 3, 1)],
                           [(0, 4, 1), (1, 5, 1), (2, 2, 1), (3, 3, 1)],
                           [(0, 6, 1), (1, 7, 1), (2, 8, 1), (3, 9, 1)]])
        y_data = np.array([0, 0, 0])
        self.ctr_engine.train(x_data, y_data)

        # Test the predict works
        self.ctr_engine.predict(np.array([(0, 0, 1), (1, 1, 1), (2, 2, 1), (3, 3, 1)]))
        self.ctr_engine.predict(np.array([(0, 4, 1), (1, 5, 1), (2, 2, 1), (3, 3, 1)]))
        self.ctr_engine.predict(np.array([(0, 6, 1), (1, 7, 1), (2, 8, 1), (3, 9, 1)]))
