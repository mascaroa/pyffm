from .ffm_engine import FFMEngine
from .fm_engine import FMEngine


class MetaFactory(type):
    ENGINE_DICT = {'ffm': FFMEngine,
                   'fm': FMEngine}

    def __getitem__(cls, item):
        return cls.ENGINE_DICT[item.lower()]

    def __contains__(cls, item):
        return item.lower() in cls.ENGINE_DICT

    def __str__(cls):
        return str(list(cls.ENGINE_DICT.keys()))


class EngineFactory(metaclass=MetaFactory):
    pass
