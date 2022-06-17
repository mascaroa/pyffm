from enum import Enum, auto


class ModelType(Enum):
    FFM = auto()
    FM = auto()


class ProblemType(Enum):
    REGRESSION = auto()
    CLASSIFICATION = auto()


class TrainingParams(Enum):
    SPLIT_FRAC = auto()
    EPOCH = auto()
    REG_LAMBDA = auto()
    SIGMOID = auto()
    PARALLEL = auto()
    EARLY_STOP = auto()


class IOParams(Enum):
    MODEL_DIR = auto()
    MODEL_FILENAME = auto()

