import sys
import numpy as np
from multiprocessing import Queue, Process
import inspect


def logistic(x):
    return np.divide(1, (1 + np.exp(-x)))


def run_as_subprocess(func):
    queue = Queue()

    if 'win' in sys.platform:  # Windows requires pickling all Process() args, so this can't be done (also isn't needed)
        return func

    def bottom_decorator(*args, **kwargs):
        in_queue = kwargs.pop('Q')
        in_queue.put(func(*args, **kwargs))

    def top_decorator(*args, **kwargs):
        attempt = 0
        timeout = kwargs.pop('timeout', None)
        retries = kwargs.pop('retries', None)
        if 'Q' in kwargs:
            return NameError(f"Can't use Q as kwarg name when using this function! {inspect.stack()[1].function}")
        while True:
            p = Process(target=bottom_decorator,
                        args=args,
                        kwargs={**kwargs, 'Q': queue})
            p.start()
            try:
                result = queue.get(timeout=timeout)
            except Exception as e:
                if p.is_alive():
                    p.terminate()
                if attempt < retries:
                    attempt += 1
                    continue
                raise e
            return result

    return top_decorator


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
