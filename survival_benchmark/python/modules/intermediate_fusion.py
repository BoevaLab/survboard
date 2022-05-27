import torch
from survival_benchmark.python.modules.autoencoder import FCBlock


class ZeroImputation(torch.nn.Module):
    def impute(self, x):
        return torch.nan_to_num(x, nan=0.0)


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


class MultiModalDropout(ZeroImputation):
    def __init__(
        self, blocks, p_multimodal_dropout=0.0, upweight=True
    ) -> None:
        super().__init__()
        self.blocks = blocks
        self.p_multimodal_dropout = p_multimodal_dropout
        self.upweight = upweight

    def multimodal_dropout(self, x):
        if self.p_multimodal_dropout > 0 and self.training:
            x = multimodal_dropout(
                x=x,
                blocks=self.blocks,
                p_multimodal_dropout=self.p_multimodal_dropout,
                upweight=self.upweight,
            )
        return x


class IntermediateFusionPoE(MultiModalDropout):
    def __init__(
        self,
        blocks,
        params,
        fusion="mean",
        beta=1,
        missing_modalities="poe",
        p_multimodal_dropout=0.0,
        upweight=True,
        alpha=0,
    ) -> None:
        super().__init__(
            blocks=blocks,
            p_multimodal_dropout=params.get("p_multimodal_dropout", 0.0),
            upweight=params.get("upweight", True),
        )
        self.block_encoder = torch.nn.ModuleList(
            [
                FCBlock(
                    params={
                        "input_size": len(blocks[i]),
                        "fc_layers": params.get("encoder_fc_layers", 1),
                        "fc_units": params.get("encoder_fc_units", [128, 128]),
                        "fc_activation": params.get(
                            "fc_activation", ["relu", "None"]
                        ),
                        "fc_batchnorm": params.get("fc_batchnorm", "True"),
                        "fc_dropout": params.get("fc_dropout", 0.5),
                    }
                )
                for i in range(len(blocks))
            ]
        )
        self.unimodal_log_hazard = torch.nn.ModuleList(
            [
                FCBlock(
                    params={
                        "input_size": params.get(
                            "encoder_fc_units", [128, 128]
                        )[-1]
                        / 2,
                        "fc_layers": params.get("hazard_fc_layers", 2),
                        "fc_units": params.get("hazard_fc_units", [32, 1]),
                        "fc_last_layer_bias": params.get(
                            "hazard_fc_last_layer_bias", False
                        ),
                        "fc_activation": params.get(
                            "fc_activation", ["relu", "None"]
                        ),
                        "fc_batchnorm": params.get("fc_batchnorm", "True"),
                        "fc_dropout": params.get("fc_dropout", 0.5),
                    }
                )
            ]
            * len(blocks)
        )
        self.joint_log_hazard = FCBlock(
            params={
                "input_size": params.get("encoder_fc_units", [128, 128])[-1]
                / 2,
                "fc_layers": params.get("hazard_fc_layers", 2),
                "fc_units": params.get("hazard_fc_units", [32, 1]),
                "fc_last_layer_bias": params.get(
                    "hazard_fc_last_layer_bias", False
                ),
                "fc_activation": params.get("fc_activation", ["relu", "None"]),
                "fc_batchnorm": params.get("fc_batchnorm", "True"),
                "fc_dropout": params.get("fc_dropout", 0.5),
            }
        )
        self.blocks = blocks
        self.params = params
        self.fusion = fusion
        self.beta = beta
        self.missing_modalities = missing_modalities
        self.p_multimodal_dropout = p_multimodal_dropout
        self.upweight = upweight
        self.alpha = alpha

    def product_of_experts(self, mask, mu, log_var):
        # variance = torch.eye(mu[0].shape[0])
        variance = torch.ones(mu[0].shape)
        for modality in range(len(mu)):
            variance += torch.stack(
                [mask[:, modality]] * log_var[modality].shape[1], axis=1
            ) * torch.div(1.0, torch.exp(log_var[modality]))
        variance = torch.div(1.0, variance)
        log_variance = torch.log(variance)

        mu_poe = torch.zeros(variance.shape)
        for modality in range(len(mu)):
            mu_poe += (
                torch.stack(
                    [mask[:, modality]] * log_var[modality].shape[1], axis=1
                )
                * torch.div(1, torch.exp(log_var[modality]))
                * mu[modality]
            )
        mu_poe = variance * mu_poe
        return mu_poe, log_variance

    def find_missing_modality_mask(self, x, blocks):
        mask = torch.zeros((x.shape[0], len(blocks)))
        for ix, modality in enumerate(blocks):
            mask[:, ix] = (
                torch.sum(torch.isnan(x[:, modality]), axis=1) != len(modality)
            ).long()
        return mask

    def forward(self, x):
        x = self.impute(x)
        mask = torch.ones((x.shape[0], len(self.blocks)))
        if self.missing_modalities == "multimodal_dropout":
            x = self.multimodal_dropout(x)
        elif self.missing_modalities == "poe":
            mask = self.find_missing_modality_mask(x, self.blocks)

        mu = []
        log_var = []
        for i in range(len(self.blocks)):
            tmp = self.block_encoder[i](x[:, self.blocks[i]])
            mu.append(
                tmp[
                    :,
                    : int(
                        self.params.get("encoder_fc_units", [128, 128])[-1] / 2
                    )
                ]
            )
            log_var.append(
                tmp[
                    :,
                    int(
                        self.params.get("encoder_fc_units", [128, 128])[-1] / 2
                    ) :
                ]
            )

        joint_mu, joint_log_var = self.product_of_experts(mask, mu, log_var)
        joint_posterior_distribution = torch.distributions.normal.Normal(
            joint_mu, torch.sqrt(torch.exp(joint_log_var))
        )
        joint_posterior = joint_posterior_distribution.rsample()
        joint_log_hazard = self.joint_log_hazard(joint_posterior)
        unimodal_posterior_distributions = [None] * len(self.blocks)
        unimodal_posteriors = [None] * len(self.blocks)
        unimodal_log_hazard = [None] * len(self.blocks)
        for ix in range(len(mu)):
            unimodal_posterior_distributions[
                ix
            ] = torch.distributions.normal.Normal(
                mu[ix], torch.sqrt(torch.exp(log_var[ix]))
            )
            unimodal_posteriors[ix] = unimodal_posterior_distributions[
                ix
            ].rsample()
            unimodal_log_hazard[ix] = self.unimodal_log_hazard[ix](
                unimodal_posteriors[ix]
            )

        return (
            joint_log_hazard,
            joint_posterior,
            joint_posterior_distribution,
            unimodal_log_hazard,
            unimodal_posteriors,
            unimodal_posterior_distributions,
        )
