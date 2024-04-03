import torch

from survboard.python.utils.misc_utils import eh_loss, negative_partial_log_likelihood


class cox_ph_criterion(torch.nn.Module):
    def forward(self, predicted, target):
        if isinstance(predicted, tuple) and len(predicted > 1):
            predicted = predicted[0]
        print(predicted)
        return negative_partial_log_likelihood(
            predicted,
            target[:, 0],
            target[:, 1],
        )


class eh_criterion(torch.nn.Module):
    def forward(self, predicted, target):
        if isinstance(predicted, tuple) and len(predicted > 1):
            predicted = predicted[0]
        return eh_loss(
            predicted,
            target[:, 0],
            target[:, 1],
        )
