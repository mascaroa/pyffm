from abc import ABC, abstractmethod


# TODO: list all these in docs more clearly
class BaseEngine(ABC):
    _save_features = ['bias', 'bias_grad', 'lin_terms', 'lin_grads', 'latent_w', 'latent_grads']

    def __init__(self, training_params):
        self.model = None
        self.epochs = training_params.pop('epochs', 10)
        self.learn_rate = training_params.pop('learn_rate', 0.2)
        self.train_quiet = training_params.pop('quiet', False)
        self.early_stop = training_params.pop('early_stop', False)
        self.parallel = training_params.pop('parallel', True)

        self._training_params = training_params
        self._training_params['reg_lambda'] = training_params.get('reg_lambda', 0.002)
        self._training_params['num_latent'] = training_params.get('num_latent', 4)
        self._training_params['sigmoid'] = training_params.get('sigmoid', False)
        self._training_params['regression'] = training_params.get('regression', False)

        self.best_loss = None
        self.best_loss_epoch = None

    @abstractmethod
    def create_model(self, *args, **kwargs):
        pass

    @abstractmethod
    def train(self, x_train, y_train, x_test, y_test):
        pass

    @abstractmethod
    def predict(self, x):
        pass

    @abstractmethod
    def set_log_level(self, log_level):
        pass

    @abstractmethod
    def save_model(self, model_path):
        pass
