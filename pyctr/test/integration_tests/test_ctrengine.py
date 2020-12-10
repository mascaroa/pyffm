import unittest
import numpy as np


from pyctr.engine.ffm_engine import FFMEngine


class TestCTREngine(unittest.TestCase):
    def setUp(self) -> None:
        self.ctr_engine = FFMEngine(training_params={'epochs': 10})

    def test_basic_train(self):
        train_data = [[1, (0, 0, 1), (1, 1, 1), (2, 2, 1), (3, 3, 1)],
                      [1, (0, 4, 1), (1, 5, 1), (2, 2, 1), (3, 3, 1)],
                      [1, (0, 6, 1), (1, 7, 1), (2, 8, 1), (3, 9, 1)]]
        self.ctr_engine.train(train_data)
        print(self.ctr_engine.predict([(0, 0, 1), (1, 1, 1), (2, 2, 1), (3, 3, 1)]))
        print(self.ctr_engine.predict([(0, 4, 1), (1, 5, 1), (2, 2, 1), (3, 3, 1)]))
        print(self.ctr_engine.predict([(0, 6, 1), (1, 7, 1), (2, 8, 1), (3, 9, 1)]))

    def test_bigger_train(self):
        self.ctr_engine = FFMEngine(training_params={'epochs': 10})
        data_in = []
        with open('small_train.txt', 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                click = [int(line.replace('\n', '').split(' ')[0])]
                features = [(int(val.split(':')[0]), int(val.split(':')[1]), float(val.split(':')[2])) for val in line.replace('\n', '').split(' ')[1:]]
                data_in.append(click + features)
        self.ctr_engine.train(data_in, x_test=data_in)
        c_m = [[0, 0], [0, 0]]
        all_pos = 0
        for row in data_in:
            pred = self.ctr_engine.predict(row[1:])
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
        pass
