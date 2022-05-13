from tokenize import group
import torch
import numpy as np
from lifelines.utils import concordance_index
from skorch.net import NeuralNet
from sksurv.linear_model.coxph import BreslowEstimator
from torch import nn

from collections.abc import Iterable

# Source: https://stackoverflow.com/questions/2158395/flatten-an-irregular-list-of-lists
def flatten(l):
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el


def transform_survival_target(time, event):
    return np.array([f"{time[i]}|{event[i]}" for i in range(len(time))])


def inverse_transform_survival_target(y):
    return (
        np.array([float(i.rsplit("|")[0]) for i in y]),
        np.array([int(i.rsplit("|")[1]) for i in y]),
    )


def inverse_transform_survival_function(y):
    return np.vstack([np.array(i.rsplit("|")) for i in y])


class BaseSurvivalNeuralNet(NeuralNet):
    def get_loss(self, y_pred, y_true, X=None, training=False):
        return self.criterion_(y_pred, y_true)


class HazardRegression(nn.Module):
    def __init__(
        self,
        input_dimension,
        n_output,
        hidden_layer_sizes,
        activation=nn.ReLU,
        p_dropout=0.0,
    ) -> None:
        super().__init__()
        hidden_layer_sizes = hidden_layer_sizes + [n_output]
        self.activation = activation
        self.p_dropout = p_dropout
        self.hidden_layer_sizes = hidden_layer_sizes
        self.n_output = n_output
        self.hazard = nn.Sequential(
            *flatten(
                [
                    nn.Linear(input_dimension, hidden_layer_sizes[0]),
                    nn.ReLU(),
                    nn.Dropout(p_dropout),
                ]
                + [
                    [
                        nn.Linear(
                            hidden_layer_sizes[i], hidden_layer_sizes[i + 1]
                        ),
                        nn.ReLU(),
                        nn.Dropout(p_dropout),
                    ]
                    for i in range(len(hidden_layer_sizes) - 1)
                ]
            )
        )


class DeepSurv(nn.Module):
    def __init__(
        self,
        input_dimension,
        hidden_layer_sizes,
        activation=nn.ReLU,
        p_dropout=0.0,
    ):
        self.log_hazard = HazardRegression(
            input_dimension, 1, hidden_layer_sizes, activation, p_dropout
        )

    def forward(self, x):
        return self.log_hazard(x)


class CoxPHNet(BaseSurvivalNeuralNet):
    def fit(self, X, y=None, **fit_params):
        if not self.warm_start or not self.initialized_:
            self.initialize()

        time, event = inverse_transform_survival_target(y)
        self.train_time = time
        self.train_event = event
        self.partial_fit(X, y, **fit_params)
        self.fit_breslow(self.forward(X), time, event)
        return self

    def fit_breslow(self, log_hazard_ratios, time, event):
        self.breslow = BreslowEstimator().fit(log_hazard_ratios, time, event)

    def predict_survival_function(self, log_hazard_ratios, times):
        survival_functions = self.breslow.get_survival_function(
            log_hazard_ratios
        )
        return np.array(
            ["|".join(i(times).astype(str)) for i in survival_functions]
        )

    def predict(self, X):
        log_hazard_ratios = self.forward(X)
        survival_function = self.predict_survival_function(
            log_hazard_ratios, self.train_time
        )
        return survival_function


class GDPNet(BaseSurvivalNeuralNet):
    def get_loss(self, y_pred, y_true, X=None, training=False):
        criterion = self.criterion_(y_pred, y_true)
        torch.tensor(0, requires_grad=True)
        for block in self.module_.blocks:
            group_lasso = torch.sum(
                torch.stack(
                    [
                        torch.sqrt(len(block))
                        * torch.norm(
                            self.module_.hazard[0].weight[:, block], p=2
                        )
                    ]
                )
            )

        weight_decay = torch.sum(
            torch.stack(
                [
                    torch.square(torch.norm(i, p=2))
                    for i in self.module_.hazard[1:].parameters()
                ]
            )
        )
        return criterion + self.module_.lambda_ * weight_decay + self.module_.alpha * group_lasso
