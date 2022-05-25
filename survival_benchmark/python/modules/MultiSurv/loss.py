"""Loss."""
import warnings
import torch
from itertools import combinations


class Loss(torch.nn.Module):
    def __init__(
        self,
        aux_criterion=None,
        is_multimodal=True,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        super(Loss, self).__init__()
        self.aux_criterion = aux_criterion
        self.is_multimodal = is_multimodal
        self.device = device

    def _convert_labels(self, time, event, breaks):
        """Convert event and time labels to label array.

        Each patient label array has dimensions number of intervals x 2:
            * First half is 1 if patient survived interval, 0 if not.
            * Second half is for non-censored times and is 1 for time interval
            in which event time falls and 0 for other intervals.
        """
        n_intervals = len(breaks) - 1
        timegap = breaks[1:] - breaks[:-1]
        breaks_midpoint = breaks[:-1] + 0.5 * timegap

        out = torch.zeros(len(time), n_intervals * 2, device=self.device)

        for i, (t, e) in enumerate(zip(time, event)):
            # t = torch.round(t * 365)  # From years to days

            if e:  # if not censored
                # survived time intervals where time >= upper limit
                out[i, 0:n_intervals] = 1.0 * (t >= breaks[1:])
                # if time is greater than end of last time interval, no
                # interval is marked
                if t < breaks[-1]:
                    # Mark first bin where survival time < upper break-point
                    idx = torch.nonzero(t < breaks[1:]).squeeze()
                    if idx.shape:  # t not in last interval
                        idx = idx[0]
                    out[i, n_intervals + idx] = 1
            else:  # if censored
                # if lived more than half-way through interval, give credit for
                # surviving the interval
                out[i, 0:n_intervals] = 1.0 * (t >= breaks_midpoint)

        return out

    def _reduction(self, loss, reduction):
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        raise ValueError(f'"reduction" must be "none", "mean" or "sum".')

    def _compute_auxiliary_loss(self, features):
        "Embedding vector distance loss."
        losses = []

        y = torch.ones(1).to(self.device)
        for x1, x2 in combinations(features, 2):
            losses.append(self.aux_criterion(x1, x2, y))

        loss = torch.tensor(losses).mean()

        return loss.to(self.device)

    def _neg_log_likelihood(self, risk, label, break_list, reduction="mean"):
        n_intervals = len(break_list) - 1

        all_patients = 1.0 + label[:, 0:n_intervals] * (risk - 1.0)
        noncensored = 1.0 - label[:, n_intervals : 2 * n_intervals] * risk

        neg_log_like = -torch.log(torch.clamp(torch.cat((all_patients, noncensored), dim=1), 1e-07, None))

        return self._reduction(neg_log_like, reduction)

    def forward(self, risk, times=None, events=None, breaks=None, modality_features=None, device=None):

        label_array = self._convert_labels(times, events, breaks).to(device)
        loss = self._neg_log_likelihood(risk, label_array.to(self.device), breaks)

        if not self.is_multimodal and self.aux_criterion is not None:
            warnings.warn("Input data is unimodal: auxiliary" + " loss is not applicable.")

        if self.is_multimodal and self.aux_criterion is not None:
            # Embedding vector distance loss
            auxiliary_loss = self._compute_auxiliary_loss(modality_features)
            loss = (1.0 * auxiliary_loss) + (0.05 * loss)

        return loss
