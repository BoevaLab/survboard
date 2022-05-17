from collections.abc import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def create_risk_matrix(observed_survival_time):
    return (
        (
            torch.outer(observed_survival_time, observed_survival_time)
            >= torch.square(observed_survival_time)
        )
        .long()
        .T
    )


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


def negative_partial_log_likelihood(
    predicted_log_hazard_ratio,
    observed_survival_time,
    observed_event_indicator,
):
    risk_matrix = create_risk_matrix(observed_survival_time)
    return torch.negative(
        torch.sum(
            (observed_event_indicator == 1)
            * (
                predicted_log_hazard_ratio
                - torch.log(
                    torch.sum(
                        risk_matrix * torch.exp(predicted_log_hazard_ratio),
                        axis=1,
                    )
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


def similarity_loss(encoded_blocks, M):
    cos = nn.CosineSimilarity(dim=0, eps=1e-08)
    loss = torch.tensor(0.0)
    n_patients = encoded_blocks[0].shape[0]
    for patient in range(n_patients):
        for matched_patient in range(n_patients):
            if patient == matched_patient:
                continue
            else:
                patient_similarity = torch.tensor(0.0)
                matched_patient_similarity = torch.tensor(0.0)
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
        cox_loss = negative_partial_log_likelihood(
            log_hazard_ratio, torch.tensor(time), torch.tensor(event)
        )
        similarity_loss_ = similarity_loss(encoded_blocks, M)
        return cox_loss + similarity_loss_


class cox_criterion(nn.Module):
    def forward(self, prediction, target):
        time, event = inverse_transform_survival_target(target)
        log_hazard_ratio = prediction
        cox_loss = negative_partial_log_likelihood(
            log_hazard_ratio, torch.tensor(time), torch.tensor(event)
        )
        return cox_loss
