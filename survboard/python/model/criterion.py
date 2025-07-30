import torch
from survboard.python.utils.misc_utils import (
    eh_loss,
    negative_partial_log_likelihood,
    nll_logistic_hazard,
)


def sparse_group_lasso(weights, lamb, alpha, groups):
    lasso_penalty = torch.norm(input=weights, p=1)
    group_lasso_penalty = 0
    for group in groups:
        group_lasso_penalty += torch.sqrt(group.shape[0]) * torch.norm(
            input=weights[group, :], p=2
        )

    return lamb * alpha * lasso_penalty + lamb * (1 - alpha) * group_lasso_penalty


class cox_ph_criterion(torch.nn.Module):
    def forward(self, predicted, target):
        if isinstance(predicted, tuple) and len(predicted > 1):
            predicted = predicted[0]
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


class cox_ph_criterion_sgl(torch.nn.Module):
    def forward(self, predicted, target, alpha, lamb, weights, groups):
        if isinstance(predicted, tuple) and len(predicted > 1):
            predicted = predicted[0]
        print(
            negative_partial_log_likelihood(
                predicted,
                target[:, 0],
                target[:, 1],
            )
        )
        print(f"Lambd: {lamb}")
        print(f"Alpha: {alpha}")
        print(target[:, 1])
        raise ValueError
        return negative_partial_log_likelihood(
            predicted,
            target[:, 0],
            target[:, 1],
        ) + sparse_group_lasso(
            weights=weights,
            lamb=lamb / torch.sum(target[:, 1]),
            alpha=alpha,
            groups=groups,
        )


class eh_criterion_sgl(torch.nn.Module):
    def forward(self, predicted, target, alpha, lamb, weights, groups):
        if isinstance(predicted, tuple) and len(predicted > 1):
            predicted = predicted[0]
        return eh_loss(
            predicted,
            target[:, 0],
            target[:, 1],
        ) + sparse_group_lasso(weights=weights, lamb=lamb, alpha=alpha, groups=groups)


class discrete_time_criterion(torch.nn.Module):
    def forward(self, predicted, target, times):
        return nll_logistic_hazard(predicted, target[:, 0], target[:, 1], times)


class discrete_time_sgl_criterion(torch.nn.Module):
    def forward(self, predicted, target, times, alpha, lamb, weights, groups):
        return nll_logistic_hazard(
            predicted, target[:, 0], target[:, 1], times
        ) + sparse_group_lasso(weights=weights, lamb=lamb, alpha=alpha, groups=groups)
