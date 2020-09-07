from typing import Union

from pyctr.engine.model.ffm_model import FFMModel
from pyctr.engine.model.fm_model import FMModel
from pyctr.engine.model.poly2_model import Poly2Model



MODEL_DICT = {'ffm': FFMModel,
              'fm': FMModel,
              'poly2': Poly2Model}

class CTREngine:
    def __init__(self, model='ffm'):
        self.model_type = model
        self.model: Union[FFMModel, FMModel, Poly2Model]

    def create_model(self, args, kwargs):
        self.model = MODEL_DICT[self.model_type](*args, *kwargs)

    def _calc_subgrad(self, x, y):
        for
        self.model.subgrad()
