import torch
import torch.nn.functional as F
from skorch.net import NeuralNet
from sksurv.linear_model.coxph import BreslowEstimator
from torch import nn

from survival_benchmark.python.utils.utils import (
    flatten,
    inverse_transform_survival_target,
)


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
                    nn.BatchNorm1d(hidden_layer_sizes[0]),
                    nn.Dropout(p_dropout),
                ]
                + [
                    [
                        nn.Linear(
                            hidden_layer_sizes[i],
                            hidden_layer_sizes[i + 1],
                            bias=bool(i == (len(hidden_layer_sizes) - 1)),
                        ),
                        nn.ReLU(),
                        nn.BatchNorm1d(hidden_layer_sizes[i + 1]),
                        nn.Dropout(p_dropout),
                    ]
                    for i in range(len(hidden_layer_sizes) - 1)
                ]
            )
        )
        self.hazard = self.hazard[:-3]

    def forward(self, X):
        return self.hazard(X)


class DeepSurv(nn.Module):
    def __init__(
        self,
        input_dimension,
        hidden_layer_sizes,
        activation=nn.ReLU,
        p_dropout=0.0,
    ):
        super().__init__()
        self.log_hazard = HazardRegression(
            input_dimension, 1, hidden_layer_sizes, activation, p_dropout
        )
        self.input_dimension = input_dimension
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.p_dropout = p_dropout

    def forward(self, x):
        return self.log_hazard(x)


class GDP(nn.Module):
    def __init__(
        self,
        blocks,
        hidden_layer_sizes,
        activation=nn.ReLU,
        p_dropout=0.0,
        scale=0.001,
        alpha=0.5,
    ):
        super().__init__()
        self.log_hazard = HazardRegression(
            sum([len(block) for block in blocks]),
            1,
            hidden_layer_sizes,
            activation,
            p_dropout,
        )
        self.blocks = blocks
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.p_dropout = p_dropout
        self.scale = scale
        self.alpha = alpha

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
        self.fit_breslow(
            self.module_.forward(torch.tensor(X)).detach().numpy().ravel(),
            time,
            event,
        )
        return self

    def fit_breslow(self, log_hazard_ratios, time, event):
        self.breslow = BreslowEstimator().fit(log_hazard_ratios, event, time)

    def predict_survival_function(self, X):
        log_hazard_ratios = self.forward(X)
        survival_function = self.breslow.get_survival_function(
            log_hazard_ratios
        )
        return survival_function

    def predict(self, X):
        log_hazard_ratios = self.forward(X)
        return log_hazard_ratios


class CheerlaEtAlNet(CoxPHNet):
    def get_loss(self, y_pred, y_true, X=None, training=False):
        criterion = self.criterion_(y_pred, y_true, self.module_.M)
        return criterion

    def fit(self, X, y=None, **fit_params):
        if not self.warm_start or not self.initialized_:
            self.initialize()

        time, event = inverse_transform_survival_target(y)
        self.train_time = time
        self.train_event = event
        self.partial_fit(X, y, **fit_params)
        self.fit_breslow(
            self.module_.forward(torch.tensor(X))[0].detach().numpy().ravel(),
            time,
            event,
        )
        return self

    def forward(self, X, training=False, device="cpu"):
        y_infer = list(self.forward_iter(X, training=training, device=device))[
            0
        ][0]
        return y_infer


class GDPNet(CoxPHNet):
    def get_loss(self, y_pred, y_true, X=None, training=False):
        criterion = self.criterion_(y_pred, y_true)
        for block in self.module_.blocks:
            group_lasso = torch.sum(
                torch.stack(
                    [
                        torch.sqrt(torch.tensor(len(block)))
                        * torch.norm(
                            self.module_.log_hazard.hazard[0].weight[:, block],
                            p=2,
                        )
                    ]
                )
            )

        lasso = torch.sum(
            torch.stack(
                [
                    torch.norm(i, p=1)
                    for i in self.module_.log_hazard.hazard[1:].parameters()
                ]
            )
        )
        return (
            criterion
            + self.module_.scale * (1 - self.module_.alpha) * lasso
            + self.module_.scale * self.module_.alpha * group_lasso
        )


# Adapted from: https://github.com/gevaertlab/MultimodalPrognosis/blob/master/modules.py
class Highway(nn.Module):
    def __init__(self, size, num_layers, f):

        super(Highway, self).__init__()

        self.num_layers = num_layers
        self.nonlinear = nn.ModuleList(
            [nn.Linear(size, size) for _ in range(num_layers)]
        )
        self.linear = nn.ModuleList(
            [nn.Linear(size, size) for _ in range(num_layers)]
        )
        self.gate = nn.ModuleList(
            [nn.Linear(size, size) for _ in range(num_layers)]
        )
        self.f = f

    def forward(self, x):
        for layer in range(self.num_layers):
            gate = torch.sigmoid(self.gate[layer](x))
            nonlinear = self.f(self.nonlinear[layer](x))
            linear = self.linear[layer](x)
            x = gate * nonlinear + (1 - gate) * linear

        return x


class cheerla_et_al_genomic_encoder(nn.Module):
    def __init__(
        self,
        input_dimension,
        encoding_dimension,
        p_dropout=0.0,
        highway_cycles=10,
    ) -> None:
        super().__init__()
        self.encode = nn.Sequential(
            nn.Linear(input_dimension, encoding_dimension),
            Highway(encoding_dimension, highway_cycles, F.relu),
            nn.Dropout(p_dropout),
            nn.Sigmoid(),
        )
        self.input_dimension = input_dimension
        self.encoding_dimension = encoding_dimension
        self.p_dropout = p_dropout
        self.highway_cycles = highway_cycles

    def forward(self, X):
        return self.encode(X)


class cheerla_et_al_clinical_encoder(nn.Module):
    def __init__(self, input_dimension, encoding_dimension=512) -> None:
        super().__init__()
        self.encode = nn.Sequential(
            nn.Linear(input_dimension, encoding_dimension), nn.Sigmoid()
        )
        self.input_dimension = input_dimension
        self.encoding_dimension = encoding_dimension

    def forward(self, X):
        return self.encode(X)


class cheerla_et_al(nn.Module):
    def __init__(
        self,
        blocks,
        encoding_dimension,
        p_dropout=0.0,
        M=0.1,
        p_multimodal_dropout=0.25,
    ) -> None:
        super().__init__()
        block_encoders = [
            cheerla_et_al_clinical_encoder(len(blocks[0]), encoding_dimension)
        ] + [
            cheerla_et_al_genomic_encoder(
                len(i), encoding_dimension, p_dropout
            )
            for i in blocks[1:]
        ]
        self.block_encoders = nn.ModuleList(block_encoders)
        self.hazard = HazardRegression(
            encoding_dimension, 1, [128], nn.Sigmoid, p_dropout
        )
        self.blocks = blocks
        self.encoding_dimension = encoding_dimension
        self.p_dropout = p_dropout
        self.M = M
        self.p_multimodal_dropout = p_multimodal_dropout

    def multimodal_dropout(self, X, p_dropout, blocks, upweight=True):
        for block in blocks:
            if not torch.all(X[:, block] == 0):
                msk = torch.where(
                    (torch.rand(X.shape[0]) <= p_dropout).long()
                )[0]
                X[:, torch.tensor(block)][msk, :] = torch.zeros(
                    X[:, torch.tensor(block)][msk, :].shape
                )

        # Subset out patients for which all modalities are missing
        msk = torch.where(
            torch.logical_not(torch.sum(X == 0, axis=1) == X.shape[0])
        )[0]
        X = X[msk, :]
        if upweight:
            X = X / (1 - p_dropout)
        return X, msk

    def forward(self, X):
        if self.p_multimodal_dropout > 0 and self.training:
            X, msk = self.multimodal_dropout(
                X, self.p_multimodal_dropout, self.blocks
            )
        else:
            msk = torch.ones(X.shape[0])
        block_encoded = [
            self.block_encoders[i](X[:, self.blocks[i]])
            for i in range(len(self.blocks))
        ]
        # Only take non-missing and non-dropped modalities into account
        # if self.training:
        joint = torch.sum(
            torch.stack(block_encoded, axis=2), axis=2
        ) / torch.sum(torch.stack(block_encoded, axis=2) != 0, axis=2)
        hazard = self.hazard(joint)
        return hazard, block_encoded, msk.long()
