from ast import Mult
import torch
from survival_benchmark.python.modules.autoencoder import FCBlock, Encoder, Decoder
from survival_benchmark.python.utils.utils import MultiModalDropout
from survival_benchmark.python.utils.utils import negative_partial_log_likelihood


class IntermediateFusionMean(MultiModalDropout):
    def __init__(
        self,
        params: dict,
        blocks,
        p_multimodal_dropout=0.0,
        upweight=True,
        fusion="mean",
        alpha=0,
        missing_modalities="impute",
    ) -> None:
        super().__init__(
            blocks=blocks,
            p_multimodal_dropout=p_multimodal_dropout,
            upweight=upweight,
        )

        # self.params = params
        self.alpha = alpha
        self.missing_modalities = missing_modalities
        self.latent_dim = params.get("latent_dim", 64)
        block_encoder = []
        params.update({"input_size": [len(i) for i in blocks]})
        for i in range(len(blocks)):
            params_mod = params.copy()
            params_mod.update({"input_size": params["input_size"][i]})

            block_encoder += [Encoder(params_mod)]

        self.block_encoder = torch.nn.ModuleList(block_encoder)

        self.unimodal_log_hazards = torch.nn.ModuleList(
            [
                FCBlock(
                    params={
                        "input_size": self.latent_dim,
                        "fc_layers": 2,
                        "fc_units": [int(self.latent_dim / 2), 1],
                        "fc_activation": ["relu", "None"],
                        "fc_batchnorm": "True",
                        "last_layer_bias": "False",
                    }
                )
            ]
            * len(blocks)
        )

        self.joint_log_hazard = FCBlock(
            params={
                "input_size": self.latent_dim,
                "fc_layers": 2,
                "fc_units": [int(self.latent_dim / 2), 1],
                "fc_activation": ["relu", "None"],
                "fc_batchnorm": "True",
                "last_layer_bias": "False",
            }
        )

        self.blocks = blocks
        self.p_multimodal_dropout = p_multimodal_dropout
        self.upweight = upweight
        self.fusion = fusion

    def forward(self, x):

        if self.missing_modalities == "multimodal_dropout":
            x = self.zero_impute(x)
            x = self.multimodal_dropout(x)
        elif self.missing_modalities == "impute":
            x = self.zero_impute(x)

        # x = self.multimodal_dropout(self.zero_impute(x))

        stacked_embedding = torch.stack(
            [self.block_encoder[i](x[:, self.blocks[i]]) for i in range(len(self.blocks))],
            axis=1,
        )
        assert stacked_embedding.shape[1] == len(self.blocks), print(stacked_embedding.shape)
        if self.fusion == "mean":
            joint_embedding = torch.mean(stacked_embedding, axis=1)
        else:
            joint_embedding = torch.max(stacked_embedding, axis=1)

        unimodal_log_hazards = []
        for i in range(len(self.blocks)):
            unimodal_log_hazards.append(self.unimodal_log_hazards[i](stacked_embedding[:, i, :]))

        joint_log_hazard = self.joint_log_hazard(joint_embedding)
        return joint_log_hazard, unimodal_log_hazards


class intermean_criterion(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, predictions, target, alpha=0):
        joint_log_hazard, unimodal_log_hazards = predictions
        time, event = target[:, 0], target[:, 1]
        unicox = 0
        for i in range(len(unimodal_log_hazards)):
            unicox += negative_partial_log_likelihood(unimodal_log_hazards[i], time, event)

        joint_cox = negative_partial_log_likelihood(joint_log_hazard, time, event)

        return joint_cox + alpha * unicox


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
                        "fc_layers": params.get("encoder_fc_layers", 2),
                        "fc_units": params.get("encoder_fc_units", [128, 128]),
                        "fc_activation": params.get("fc_activation", ["relu", "None"]),
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
                        "input_size": params.get("encoder_fc_units", [128, 128])[-1] / 2,
                        "fc_layers": params.get("hazard_fc_layers", 2),
                        "fc_units": params.get("hazard_fc_units", [32, 1]),
                        "fc_last_layer_bias": params.get("hazard_fc_last_layer_bias", False),
                        "fc_activation": params.get("fc_activation", ["relu", "None"]),
                        "fc_batchnorm": params.get("fc_batchnorm", "True"),
                        "fc_dropout": params.get("fc_dropout", 0.5),
                        "last_layer_bias": params.get("hazard_fc_last_layer_bias", "False"),
                    }
                )
            ]
            * len(blocks)
        )
        self.joint_log_hazard = FCBlock(
            params={
                "input_size": params.get("encoder_fc_units", [128, 128])[-1] / 2,
                "fc_layers": params.get("hazard_fc_layers", 2),
                "fc_units": params.get("hazard_fc_units", [32, 1]),
                "last_layer_bias": params.get("hazard_fc_last_layer_bias", "False"),
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
            variance += torch.stack([mask[:, modality]] * log_var[modality].shape[1], axis=1) * torch.div(
                1.0, torch.exp(log_var[modality])
            )
        variance = torch.div(1.0, variance)
        log_variance = torch.log(variance)

        mu_poe = torch.zeros(variance.shape)
        for modality in range(len(mu)):
            mu_poe += (
                torch.stack([mask[:, modality]] * log_var[modality].shape[1], axis=1)
                * torch.div(1, torch.exp(log_var[modality]))
                * mu[modality]
            )
        mu_poe = variance * mu_poe
        return mu_poe, log_variance

    def find_missing_modality_mask(self, x, blocks):
        mask = torch.zeros((x.shape[0], len(blocks)))
        for ix, modality in enumerate(blocks):
            mask[:, ix] = (torch.sum(torch.isnan(x[:, modality]), axis=1) != len(modality)).long()
        return mask

    def forward(self, x):
        # x = self.zero_impute(x)
        mask = torch.ones((x.shape[0], len(self.blocks)))
        if self.missing_modalities == "multimodal_dropout":
            x = self.zero_impute(x)
            x = self.multimodal_dropout(x)
        elif self.missing_modalities == "impute":
            x = self.zero_impute(x)
        elif self.missing_modalities == "poe":
            mask = self.find_missing_modality_mask(x, self.blocks)
            x = self.zero_impute(x)

        mu = []
        log_var = []
        for i in range(len(self.blocks)):
            tmp = self.block_encoder[i](x[:, self.blocks[i]])
            mu.append(tmp[:, : int(self.params.get("encoder_fc_units", [128, 128])[-1] / 2)])
            log_var.append(tmp[:, int(self.params.get("encoder_fc_units", [128, 128])[-1] / 2) :])

        joint_mu, joint_log_var = self.product_of_experts(mask, mu, log_var)
        joint_posterior_distribution = torch.distributions.normal.Normal(joint_mu, torch.sqrt(torch.exp(joint_log_var)))
        joint_posterior = joint_posterior_distribution.rsample()
        joint_log_hazard = self.joint_log_hazard(joint_posterior)
        unimodal_posterior_distributions = [None] * len(self.blocks)
        unimodal_posteriors = [None] * len(self.blocks)
        unimodal_log_hazard = [None] * len(self.blocks)
        for ix in range(len(mu)):
            unimodal_posterior_distributions[ix] = torch.distributions.normal.Normal(
                mu[ix], torch.sqrt(torch.exp(log_var[ix]))
            )
            unimodal_posteriors[ix] = unimodal_posterior_distributions[ix].rsample()
            unimodal_log_hazard[ix] = self.unimodal_log_hazard[ix](unimodal_posteriors[ix])

        return (
            joint_log_hazard,
            joint_posterior,
            joint_posterior_distribution,
            unimodal_log_hazard,
            unimodal_posteriors,
            unimodal_posterior_distributions,
        )
