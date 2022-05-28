import torch
from survival_benchmark.python.utils.utils import (
    negative_partial_log_likelihood,
)


class intermediate_fusion_poe_criterion(torch.nn.Module):
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


class intermediate_fusion_mean_criterion(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, predictions, target, alpha=0):
        joint_log_hazard, unimodal_log_hazards = predictions
        time, event = target[:, 0], target[:, 1]
        unicox = 0
        for i in range(len(unimodal_log_hazards)):
            unicox += negative_partial_log_likelihood(
                unimodal_log_hazards[i], time, event
            )

        joint_cox = negative_partial_log_likelihood(
            joint_log_hazard, time, event
        )

        return joint_cox + alpha * unicox


class dae_criterion(torch.nn.Module):
    def forward(self, predicted, target, alpha):
        time, event = target[:, 0], target[:, 1]
        cox_loss = negative_partial_log_likelihood(predicted[0], time, event)
        reconstruction_loss = torch.nn.MSELoss()(predicted[1], predicted[2])
        return alpha * cox_loss + reconstruction_loss