import torch
from skorch.callbacks import Callback
from itertools import accumulate
import torch.nn.functional as F


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


def weird_cox_loss(hazard, days_to_death, vital_status):
    _, idx = torch.sort(days_to_death)
    hazard_probs = F.softmax(
        hazard[idx].squeeze()[1 - vital_status.byte()], dim=0
    )
    hazard_cum = torch.stack(
        [torch.tensor(0.0)] + list(accumulate(hazard_probs))
    )
    N = hazard_probs.shape[0]
    weights_cum = torch.range(1, N)
    p, q = hazard_cum[1:], 1 - hazard_cum[:-1]
    w1, w2 = weights_cum, N - weights_cum
    probs = torch.stack([p, q], dim=1)
    logits = torch.log(probs)
    ll1 = (F.nll_loss(logits, torch.zeros(N).long(), reduce=False) * w1) / N
    ll2 = (F.nll_loss(logits, torch.ones(N).long(), reduce=False) * w2) / N
    loss2 = torch.mean(ll1 + ll2)
    return loss2
