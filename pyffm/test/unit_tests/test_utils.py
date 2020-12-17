import unittest
import numpy as np

import string

from pyffm.util import Map


class TestMap(unittest.TestCase):
    def test_basic(self):
        map1 = Map()
        map_size_to_test = 1000
        all_letters = string.ascii_uppercase + string.ascii_lowercase
        counter = 0
        for char in ''.join(all_letters[np.random.choice(len(all_letters))] for _ in range(map_size_to_test)):
            if char not in map1:
                counter += 1
            map_index = map1.add(char)
            self.assertEqual(map_index, map1._map_dict[char])
        self.assertEqual(len(map1), counter)
