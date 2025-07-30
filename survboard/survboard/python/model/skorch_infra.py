import pandas as pd
import torch
from skorch.callbacks import Callback
from skorch.net import NeuralNet
from skorch.utils import to_tensor
from sksurv.linear_model.coxph import BreslowEstimator
from survboard.python.utils.misc_utils import (
    InterpolateLogisticHazard,
    seed_torch,
    transform_back,
)


class BaseSurvivalNet(NeuralNet):
    def predict(self, X):
        log_hazard_ratios = self.forward(X)
        return log_hazard_ratios


class BaseSurvivalSGLNet(BaseSurvivalNet):
    def get_loss(self, y_pred, y_true, X=None, training=False):
        y_true = to_tensor(y_true, device=self.device)
        if not training:
            return self.criterion_(
                y_pred,
                y_true,
                self.module_.alpha,
                0,
                self.module_.log_hazard.hazard[0].weight,
                self.module_.blocks,
            )
        else:
            return self.criterion_(
                y_pred,
                y_true,
                self.module_.alpha,
                self.module_.lamb,
                self.module_.log_hazard.hazard[0].weight,
                self.module_.blocks,
            )


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


class CoxPHNeuralSGLNet(BaseSurvivalSGLNet):
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


class EHNeuralSGLNet(BaseSurvivalSGLNet):
    def fit(self, X, y=None, **fit_params):
        if not self.warm_start or not self.initialized_:
            self.initialize()
        self.partial_fit(X, y, **fit_params)
        return self


# Adapted from: https://github.com/havakv/pycox/blob/master/pycox/models/logistic_hazard.py
class DiscreteTimeNeuralNet(BaseSurvivalNet):
    def fit(self, X, y=None, **fit_params):
        if not self.warm_start or not self.initialized_:
            self.initialize()
        self.partial_fit(X, y, **fit_params)
        return self

    def predict_survival_function(
        self, X, duration_index, epsilon=1e-7, scheme="const_pdf", sub=10
    ):
        interpol = InterpolateLogisticHazard(
            model=self,
            scheme=scheme,
            duration_index=duration_index,
            sub=sub,
            epsilon=epsilon,
        )
        # surv = self.predict_surv(X)
        # surv = pd.DataFrame(surv, columns=duration_index)
        surv = interpol.predict_surv_df(X)
        return surv

    # def get_loss(self, y_pred, y_true, X=None, training=False):
    #     y_true = to_tensor(y_true, device=self.device)
    #     return self.criterion_(
    #         y_pred,
    #         y_true,
    #         self.module_.times,
    #     )

    def predict_hazard(
        self,
        X,
    ):
        hazard = self.forward(X).sigmoid()
        return hazard

    def predict_surv(
        self,
        X,
        epsilon=1e-7,
    ):
        hazard = self.predict_hazard(X)
        surv = (1 - hazard).add(epsilon).log().cumsum(1).exp()
        return surv


# Adapted from: https://github.com/havakv/pycox/blob/master/pycox/models/logistic_hazard.py
class DiscreteTimeNeuraSGLNet(BaseSurvivalSGLNet):
    def fit(self, X, y=None, **fit_params):
        if not self.warm_start or not self.initialized_:
            self.initialize()
        self.partial_fit(X, y, **fit_params)
        return self

    def predict_survival_function(
        self, X, duration_index, epsilon=1e-7, scheme="const_pdf", sub=10
    ):
        interpol = InterpolateLogisticHazard(
            model=self,
            scheme=scheme,
            duration_index=duration_index,
            sub=sub,
            epsilon=epsilon,
        )
        # surv = self.predict_surv(X)
        # surv = pd.DataFrame(surv, columns=duration_index)
        surv = interpol.predict_surv_df(X)
        return surv

    # def get_loss(self, y_pred, y_true, X=None, training=False):
    #     y_true = to_tensor(y_true, device=self.device)
    #     return self.criterion_(
    #         y_pred,
    #         y_true,
    #         self.module_.times,
    #     )

    def predict_hazard(
        self,
        X,
    ):
        hazard = self.forward(X).sigmoid()
        return hazard

    def predict_surv(
        self,
        X,
        epsilon=1e-7,
    ):
        hazard = self.predict_hazard(X)
        surv = (1 - hazard).add(epsilon).log().cumsum(1).exp()
        return surv


class FixSeed(Callback):
    def __init__(self, generator):
        self.generator = generator

    def initialize(self):
        seed = self.generator.integers(low=0, high=262144, size=1)[0]
        seed_torch(seed)
        return super().initialize()
