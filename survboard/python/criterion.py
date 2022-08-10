import torch

from survboard.python.utils.utils import neg_par_log_likelihood


class intermediate_fusion_mean_criterion(torch.nn.Module):
    def forward(
        self,
        predictions,
        target,
        alpha=0,
    ):
        joint_log_hazard, unimodal_log_hazards = predictions
        time, event = target[:, 0], target[:, 1]
        unicox = 0
        for i in range(len(unimodal_log_hazards)):
            unicox += neg_par_log_likelihood(
                unimodal_log_hazards[i],
                torch.unsqueeze(time, 1).float(),
                torch.unsqueeze(event, 1).float(),
            )
        joint_cox = neg_par_log_likelihood(
            joint_log_hazard,
            torch.unsqueeze(time, 1).float(),
            torch.unsqueeze(event, 1).float(),
        )
        return joint_cox + alpha * unicox


class naive_neural_criterion(torch.nn.Module):
    def forward(self, predicted, target):
        time, event = target[:, 0], target[:, 1]
        cox_loss = neg_par_log_likelihood(
            predicted,
            torch.unsqueeze(time, 1).float(),
            torch.unsqueeze(event, 1).float(),
        )
        return cox_loss
