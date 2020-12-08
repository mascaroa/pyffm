import numpy as np


def logistic(x):
    return np.divide(1, (1 + np.exp(-x)))


class Map:
    def __init__(self):
        self._map_dict = {}
        self._inv_map_dict = {}
        self._counter = 0

    def add(self, item):
        item = str(item)
        self._add_item(item)
        return self[item]

    def get(self, item, default=None):
        item = str(item)
        if item in self._inv_map_dict:
            return self._inv_map_dict[item]
        return default

    def max(self):
        return max(self._map_dict.keys())

    def _add_item(self, item):
        if item not in self._inv_map_dict:
            while self._counter in self._inv_map_dict:
                self._counter += 1
            self._map_dict[self._counter] = item
            self._inv_map_dict[item] = int(self._counter)
            self._counter += 1
        return self._inv_map_dict[item]

    def __getitem__(self, item):
        item = str(item)
        if item in self._inv_map_dict:
            return self._inv_map_dict[item]
        return False

    def __contains__(self, item):
        item = str(item)
        return item in self._inv_map_dict

    def __str__(self):
        return f'{self.__class__.__name__}[{self._map_dict}]'

    def __len__(self):
        return len(self._inv_map_dict)
