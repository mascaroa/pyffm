import unittest
import numpy as np

from pyctr import PyCTR


class TestCTREngine(unittest.TestCase):
    def setUp(self) -> None:
        self.pyctr = PyCTR(model="ffm")

    def test_basic_train(self):

        import os

        dir = os.getcwd()
        pyctr.train(os.path.join(dir, 'test/integration_tests/small_train.txt'))
        data_in = []
        with open(os.path.join(dir, 'test/integration_tests/small_train.txt'), 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                click = [int(line.replace('\n', '').split(' ')[0])]
                features = [(int(val.split(':')[0]), int(val.split(':')[1]), float(val.split(':')[2])) for val in line.replace('\n', '').split(' ')[1:]]
                data_in.append(click + features)
        c_m = [[0, 0], [0, 0]]
        all_pos = 0
        for row in data_in:
            pred = pyctr.predict(row[1:])
            if row[0] == 1 and row[0] == np.round(pred):
                c_m[0][0] += 1
            elif row[0] == 0 and row[0] == np.round(pred):
                c_m[1][1] += 1
            elif row[0] == 1 and row[0] != np.round(pred):
                c_m[1][0] += 1
            elif row[0] == 0 and row[0] != np.round(pred):
                c_m[0][1] += 1
            if row[0] == 1:
                all_pos += 1
            print(f'{row[0]} - {int(np.round(pred))}')
        print(f' {c_m[0][0]} | {c_m[0][1]} \n---------- \n  {c_m[1][0]} | {c_m[1][1]}')