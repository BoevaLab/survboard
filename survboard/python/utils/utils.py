import os
import random
from collections.abc import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from skorch.dataset import ValidSplit, get_len
from skorch.utils import to_numpy
from torch import nn

from survival_benchmark.python.utils.hyperparameters import (
    ACTIVATION_FN_FACTORY,
)


def multimodal_dropout(x, p_multimodal_dropout, blocks, upweight=True):
    for block in blocks:
        if not torch.all(x[:, block] == 0):
            msk = torch.where(
                (torch.rand(x.shape[0]) <= p_multimodal_dropout).long()
            )[0]
            x[:, torch.tensor(block)][msk, :] = torch.zeros(
                x[:, torch.tensor(block)][msk, :].shape
            )

    if upweight:
        x = x / (1 - p_multimodal_dropout)
    return x


class MultiModalDropout(torch.nn.Module):
    def __init__(
        self, blocks, p_multimodal_dropout=0.0, upweight=True
    ) -> None:
        super().__init__()
        self.blocks = blocks
        self.p_multimodal_dropout = p_multimodal_dropout
        self.upweight = upweight

    def zero_impute(self, x):
        return torch.nan_to_num(x, nan=0.0)

    def multimodal_dropout(self, x):
        if self.p_multimodal_dropout > 0 and self.training:
            x = multimodal_dropout(
                x=x,
                blocks=self.blocks,
                p_multimodal_dropout=self.p_multimodal_dropout,
                upweight=self.upweight,
            )
        return x


def create_risk_matrix(observed_survival_time):
    observed_survival_time = observed_survival_time.squeeze()
    return (
        (
            torch.outer(observed_survival_time, observed_survival_time)
            >= torch.square(observed_survival_time)
        )
        .long()
        .T
    )


class StratifiedSurvivalKFold(StratifiedKFold):
    """Adapt `StratifiedKFold` to make it usable with our adapted
    survival target string format.

    For further documentation, please refer to the `StratifiedKFold`
    documentation, as the only changes made were to adapt the string
    target format.
    """

    def _make_test_folds(self, X, y=None):
        if y is not None and isinstance(y, np.ndarray):
            # Handle string target by selecting out only the event
            # to stratify on.
            if y.dtype not in [np.dtype("float32"), np.dtype("int")]:
                y = np.array([str.rsplit(i, "|")[1] for i in y]).astype(
                    np.float32
                )

        return super()._make_test_folds(X=X, y=y)

    def _iter_test_masks(self, X, y=None, groups=None):
        if y is not None and isinstance(y, np.ndarray):
            # Handle string target by selecting out only the event
            # to stratify on.

            if y.dtype not in [np.dtype("float32"), np.dtype("int")]:
                y = np.array([str.rsplit(i, "|")[1] for i in y]).astype(
                    np.float32
                )
            else:
                event = y[:, 1]
                return super()._iter_test_masks(X, y=event)
        return super()._iter_test_masks(X, y=y)

    def split(self, X, y, groups=None):
        return super().split(X=X, y=y, groups=groups)


class StratifiedSkorchSurvivalSplit(ValidSplit):
    """Adapt `ValidSplit` to make it usable with our adapted
    survival target string format.

    For further documentation, please refer to the `ValidSplit`
    documentation, as the only changes made were to adapt the string
    target format.
    """

    def __call__(self, dataset, y=None, groups=None):
        if y is not None:
            # Handle string target by selecting out only the event
            # to stratify on.
            if y.dtype not in [np.dtype("float32"), np.dtype("int")]:
                y = np.array([str.rsplit(i, "|")[1] for i in y]).astype(
                    np.float32
                )

        bad_y_error = ValueError(
            "Stratified CV requires explicitly passing a suitable y."
        )

        if (y is None) and self.stratified:
            raise bad_y_error

        cv = StratifiedKFold(n_splits=self.cv, random_state=42, shuffle=True)

        # pylint: disable=invalid-name
        len_dataset = get_len(dataset)
        if y is not None:
            len_y = get_len(y)
            if len_dataset != len_y:
                raise ValueError(
                    "Cannot perform a CV split if dataset and y "
                    "have different lengths."
                )

        args = (np.arange(len_dataset),)
        if self._is_stratified(cv):
            args = args + (to_numpy(y),)

        idx_train, idx_valid = next(iter(cv.split(*args, groups=groups)))
        dataset_train = torch.utils.data.Subset(dataset, idx_train)
        dataset_valid = torch.utils.data.Subset(dataset, idx_valid)
        return dataset_train, dataset_valid


# Adapted from https://github.com/pytorch/pytorch/issues/7068.
def seed_torch(seed=42):
    """Sets all seeds within torch and adjacent libraries.

    Args:
        seed: Random seed to be used by the seeding functions.

    Returns:
        None
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return None


# Source: https://stackoverflow.com/questions/2158395/flatten-an-irregular-list-of-lists
def flatten(nested_list):
    for el in nested_list:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el


def transform_survival_target(time, event):
    return np.array([f"{time[i]}|{event[i]}" for i in range(len(time))])


def inverse_transform_survival_target(y):
    return (
        np.array([int(i.rsplit("|")[0]) for i in y]),
        np.array([int(i.rsplit("|")[1]) for i in y]),
    )


def inverse_transform_survival_function(y):
    return np.vstack([np.array(i.rsplit("|")) for i in y])


def negative_partial_log_likelihood_loss(
    y_true,
    y_pred,
):
    (
        observed_survival_time,
        observed_event_indicator,
    ) = inverse_transform_survival_target(y_true)
    return neg_par_log_likelihood(
        y_pred,
        torch.unsqueeze(torch.tensor(observed_survival_time), 1).float(),
        torch.unsqueeze(torch.tensor(observed_event_indicator), 1).float(),
    )


def negative_partial_log_likelihood(
    predicted_log_hazard_ratio,
    observed_survival_time,
    observed_event_indicator,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    observed_event_indicator = observed_event_indicator.to(device)
    observed_survival_time = observed_survival_time.to(device)
    predicted_log_hazard_ratio = predicted_log_hazard_ratio.to(device)
    risk_matrix = create_risk_matrix(observed_survival_time).to(device)
    return -torch.sum(
        observed_event_indicator.float().squeeze()
        * (
            predicted_log_hazard_ratio.squeeze()
            - torch.log(
                torch.sum(
                    risk_matrix.float()
                    * torch.exp(predicted_log_hazard_ratio.squeeze()),
                    axis=1,
                )
            )
        )
    ) / torch.sum(observed_event_indicator)


def has_missing_modality(encoded_blocks, patient, matched_patient, modality):
    return torch.all(
        encoded_blocks[modality][patient]
        == encoded_blocks[modality][patient][0]
    ) or torch.all(
        encoded_blocks[modality][matched_patient]
        == encoded_blocks[modality][matched_patient][0]
    )


def neg_par_log_likelihood(pred, survival_time, survival_event, cuda=0):
    """
    Calculate the average Cox negative partial log-likelihood
    Input:
        pred: linear predictors from trained model.
        survival_time: survival time from ground truth
        survival_event: survival event from ground truth: 1 for event and 0 for censored
    Output:
        cost: the survival cost to be minimized
    """
    n_observed = survival_event.sum(0)
    if n_observed <= 0:
        # Return arbitary high value for batches in which there were no events
        return torch.tensor(5.0, requires_grad=True)
    R_matrix = create_risk_matrix(survival_time).float()
    if cuda:
        R_matrix = R_matrix.cuda()
    risk_set_sum = R_matrix.mm(torch.exp(pred))
    diff = pred - torch.log(risk_set_sum)
    sum_diff_in_observed = torch.transpose(diff, 0, 1).mm(survival_event)
    loss = (-(sum_diff_in_observed) / n_observed).reshape((-1,))
    return loss


def get_blocks(feature_names):
    column_types = (
        pd.Series(feature_names).str.rsplit("_").apply(lambda x: x[0]).values
    )
    return [
        np.where(
            modality
            == pd.Series(feature_names)
            .str.rsplit("_")
            .apply(lambda x: x[0])
            .values
        )[0].tolist()
        for modality in [
            q
            for q in ["clinical", "gex", "cnv", "rppa", "mirna", "mut", "meth"]
            if q in np.unique(column_types)
        ]
    ]


class FCBlock(nn.Module):
    """Generalisable DNN module to allow for flexibility in architecture."""

    def __init__(self, params: dict) -> None:
        """Constructor.

        Args:
            params (dict): DNN parameter dictionary with the following keys:
                input_size (int): Input tensor dimensions.
                fc_layers (int): Number of fully connected layers to add.
                fc_units (List[(int)]): List of hidden units for each layer.
                fc_activation (str): Activation function to apply after each
                    fully connected layer. See utils/hyperparameter.py
                    for options.
                fc_batchnorm (bool): Whether to include a batchnorm layer.
                fc_dropout (float): Probability of dropout applied after eacch layer.
                last_layer_bias (bool): True if bias should be applied in the last layer, False otherwise. Default True.

        """
        super(FCBlock, self).__init__()

        self.input_size = params.get("input_size", 256)
        self.latent_dim = params.get("latent_dim", 64)

        self.hidden_size = params.get("fc_units", [128, 64])
        self.layers = params.get("fc_layers", 2)
        self.scaling_factor = params.get("scaling_factor", 0.5)

        self.activation = params.get("fc_activation", ["relu", "None"])
        self.dropout = params.get("fc_dropout", 0.5)
        self.batchnorm = eval(params.get("fc_batchnorm", "True"))
        self.bias_last = eval(params.get("last_layer_bias", "True"))
        bias = [True] * (self.layers - 1) + [self.bias_last]

        if (
            len(self.hidden_size) != self.layers
            and self.scaling_factor is not None
        ):
            hidden_size_generated = [self.input_size]
            # factor = (self.reduction_factor - 1) / self.reduction_factor
            for layer in range(self.layers):
                try:
                    hidden_size_generated.append(self.hidden_size[layer])
                except IndexError:
                    if layer == self.layers - 1:
                        hidden_size_generated.append(self.latent_dim)
                    else:
                        hidden_size_generated.append(
                            int(
                                hidden_size_generated[-1] * self.scaling_factor
                            )
                        )

            self.hidden_size = hidden_size_generated[1:]

        if len(self.activation) != self.layers:
            if len(self.activation) == 2:
                first, last = self.activation
                self.activation = [first] * (self.layers - 1) + [last]

            elif len(self.activation) == 1:
                self.activation = self.activation * self.layers

            else:
                raise ValueError

        modules = []
        self.hidden_units = [self.input_size] + self.hidden_size
        for layer in range(self.layers):
            modules.append(
                nn.Linear(
                    int(self.hidden_units[layer]),
                    int(self.hidden_units[layer + 1]),
                    bias=bias[layer],
                )
            )
            if self.activation[layer] != "None":
                modules.append(ACTIVATION_FN_FACTORY[self.activation[layer]])
            if self.dropout > 0:
                if layer < self.layers - 1:
                    modules.append(nn.Dropout(self.dropout))
            if self.batchnorm:
                if layer < self.layers - 1:
                    modules.append(
                        nn.BatchNorm1d(self.hidden_units[layer + 1])
                    )
        self.model = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passes input through a feed forward neural network.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size,*,input_size]

        Returns:
            torch.Tensor: Output tensor of shape [batch_size,*, hidden_sizes[-1]].
        """

        return self.model(x)
