import torch
from skorch.callbacks import Callback
from skorch.net import NeuralNet
from sksurv.linear_model.coxph import BreslowEstimator

from survboard.python.utils.misc_utils import seed_torch, transform_back


class BaseSurvivalNet(NeuralNet):
    def predict(self, X):
        log_hazard_ratios = self.forward(X)
        return log_hazard_ratios


class CoxPHNeuralNet(BaseSurvivalNet):
    def fit(self, X, y=None, **fit_params):
        if not self.warm_start or not self.initialized_:
            self.initialize()
        time, event = transform_back(y)
        self.train_time = time
        self.train_event = event
        self.partial_fit(X, y, **fit_params)
        self.fit_breslow(
            self.module_.forward(torch.tensor(X))
            .detach()
            .numpy()
            .ravel()
            .astype(float),
            time,
            event,
        )
        return self

    def fit_breslow(self, log_hazard_ratios, time, event):
        self.breslow = BreslowEstimator().fit(log_hazard_ratios, event, time)

    def predict_survival_function(self, X):
        log_hazard_ratios = self.forward(X).detach().numpy().ravel().astype(float)
        survival_function = self.breslow.get_survival_function(log_hazard_ratios)
        return survival_function


class EHNeuralNet(BaseSurvivalNet):
    def fit(self, X, y=None, **fit_params):
        if not self.warm_start or not self.initialized_:
            self.initialize()
        self.partial_fit(X, y, **fit_params)
        return self


class FixSeed(Callback):
    def __init__(self, generator):
        self.generator = generator

    def initialize(self):
        seed = self.generator.integers(low=0, high=262144, size=1)[0]
        seed_torch(seed)
        return super().initialize()
