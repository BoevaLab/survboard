import torch
from skorch.callbacks import Callback


def create_risk_matrix(observed_survival_time):
    return (
        (
            torch.outer(observed_survival_time, observed_survival_time)
            >= torch.square(observed_survival_time)
        )
        .long()
        .T
    )


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