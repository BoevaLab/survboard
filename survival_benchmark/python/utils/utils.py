from ntpath import join
import random
import os

import pandas as pd
from collections.abc import Iterable

from skorch.callbacks import Callback
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch import nn
from skorch.utils import to_numpy
from skorch.dataset import get_len, ValidSplit
from sklearn.model_selection import StratifiedKFold


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

        cv = self.check_cv(y)
        if self.stratified and not self._is_stratified(cv):
            raise bad_y_error

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
        np.array([float(i.rsplit("|")[0]) for i in y]),
        np.array([int(i.rsplit("|")[1]) for i in y]),
    )


def inverse_transform_survival_function(y):
    return np.vstack([np.array(i.rsplit("|")) for i in y])


def negative_partial_log_likelihood_loss(
    y,
    predicted_log_hazard_ratio,
    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ),
):
    (
        observed_survival_time,
        observed_event_indicator,
    ) = inverse_transform_survival_target(y)
    observed_survival_time = torch.tensor(observed_survival_time).to(device)
    observed_event_indicator = torch.tensor(observed_event_indicator).to(
        device
    )
    return negative_partial_log_likelihood(
        predicted_log_hazard_ratio,
        observed_survival_time,
        observed_event_indicator,
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


def similarity_loss(
    encoded_blocks,
    M,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    cos = nn.CosineSimilarity(dim=0, eps=1e-08)
    loss = torch.tensor(0.0, device=device)
    n_patients = encoded_blocks[0].shape[0]
    for patient in range(n_patients):
        for matched_patient in range(n_patients):
            if patient == matched_patient:
                continue
            else:
                patient_similarity = torch.tensor(0.0, device=device)
                matched_patient_similarity = torch.tensor(0.0, device=device)
                for modality in range(len(encoded_blocks)):
                    if has_missing_modality(
                        encoded_blocks, patient, matched_patient, modality
                    ):
                        pass
                    else:
                        patient_similarity += cos(
                            encoded_blocks[modality][patient],
                            encoded_blocks[modality][patient],
                        )
                        matched_patient_similarity += cos(
                            encoded_blocks[modality][patient],
                            encoded_blocks[modality][matched_patient],
                        )
            loss += F.relu(M - matched_patient_similarity + patient_similarity)
    return loss


class cheerla_et_al_criterion(nn.Module):
    def forward(self, prediction, target, M):
        time, event = inverse_transform_survival_target(target)
        log_hazard_ratio = prediction[0]
        encoded_blocks = prediction[1]
        msk = prediction[2]
        cox_loss = negative_partial_log_likelihood(
            # Take out masked patients for which all modalties were missing
            log_hazard_ratio,
            torch.tensor(time)[msk],
            torch.tensor(event)[msk],
        )
        # Not necessary to exclude masked patients since they were already
        # excluded in the forward.
        similarity_loss_ = similarity_loss(encoded_blocks, M)
        return cox_loss + similarity_loss_


def neg_par_log_likelihood(pred, survival_time, survival_event, cuda=1):
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
    R_matrix = create_risk_matrix(survival_time).float()
    R_matrix = torch.Tensor(R_matrix)
    if cuda:
        R_matrix = R_matrix.cuda()
    risk_set_sum = R_matrix.mm(torch.exp(pred))
    diff = pred - torch.log(risk_set_sum)
    sum_diff_in_observed = torch.transpose(diff, 0, 1).mm(survival_event)
    loss = (-(sum_diff_in_observed) / n_observed).reshape((-1,))
    return loss


class cox_criterion(nn.Module):
    def forward(self, prediction, target):
        time, event = inverse_transform_survival_target(target)
        log_hazard_ratio = prediction
        cox_loss = negative_partial_log_likelihood(
            log_hazard_ratio, torch.tensor(time), torch.tensor(event)
        )
        return cox_loss


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


# Adapted from https://github.com/skorch-dev/skorch/issues/280
class FixRandomSeed(Callback):
    """Ensure reproducibility within skorch by setting all seeds.

    Attributes:
        seed: Random seed to be used by the seeding functions.

    """

    def __init__(self, seed=42):
        self.seed = seed

    def initialize(self):
        seed_torch(self.seed)


class poe_criterion(torch.nn.Module):
    def forward(
        self,
        predicted,
        target,
        alpha,
        beta,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        std_normal = torch.distributions.normal.Normal(0, 1)
        time = target[:, 0]
        event = target[:, 1]
        cox_joint = negative_partial_log_likelihood(
            predicted[0], time, event
        ).to(device)
        cox_modality = [
            negative_partial_log_likelihood(log_hazard, time, event).to(device)
            for log_hazard in predicted[3]
        ]
        joint_kl = torch.div(
            torch.sum(
                torch.distributions.kl.kl_divergence(predicted[2], std_normal)
            ),
            predicted[0].shape[0],
        ).to(device)

        modality_kl = [
            torch.div(
                torch.sum(
                    torch.distributions.kl.kl_divergence(
                        posterior, std_normal
                    ).to(device)
                ),
                predicted[0].shape[0],
            )
            for posterior in predicted[5]
        ]
        return (
            cox_joint
            + beta * joint_kl
            + alpha
            * (
                torch.sum(torch.stack(cox_modality))
                + beta * torch.sum(torch.stack(modality_kl))
            )
        )
