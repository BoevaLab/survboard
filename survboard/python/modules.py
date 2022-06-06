import torch

from survboard.python.autoencoder import Decoder, Encoder, FCBlock
from survboard.python.utils.utils import MultiModalDropout


class DAE(MultiModalDropout):
    def __init__(
        self,
        params,
        blocks,
        missing_modalities="impute",
        p_multimodal_dropout=0.0,
        noise_factor=1,
        alpha=0.01,
        upweight=True,
        p_dropout=0.0,
    ) -> None:
        super().__init__(
            blocks=blocks,
            p_multimodal_dropout=p_multimodal_dropout,
            upweight=upweight,
        )

        self.alpha = alpha
        self.noise_factor = noise_factor
        self.missing_modalities = missing_modalities

        self.input_size = sum([len(i) for i in blocks])
        self.latent_dim = params.get("latent_dim")
        self.hidden_units = params.get("fc_units")
        self.params = params
        self.params["fc_dropout"] = p_dropout
        params_encoder = self.params.copy()
        params_decoder = self.params.copy()
        params_encoder.update({"input_size": self.input_size})
        self.encoder = Encoder(params_encoder)

        params_decoder.update(
            {
                "input_size": self.latent_dim,
                "latent_dim": self.input_size,
                "fc_units": self.hidden_units[-2::-1],
                "scaling_factor": 2,
            }
        )
        self.decoder = Decoder(params_decoder)
        fc_params = {
            "input_size": self.latent_dim,
            "latent_dim": 1,
            "last_layer_bias": self.params.get("hazard_fc_last_layer_bias"),
            "fc_layers": self.params.get("fc_layers"),
            "fc_units": self.params.get("hazard_fc_units"),
            "fc_dropout": self.params.get("fc_dropout"),
        }
        self.hazard = FCBlock(fc_params)

    def forward(self, x):
        if self.training:
            x = self.zero_impute(x)
            x_dropout = x
            if self.missing_modalities == "multimodal_dropout":
                x_dropout = self.multimodal_dropout(x)
            x_noisy = x_dropout + (
                self.noise_factor * torch.randn(size=x_dropout.shape)
            )
        else:
            x = self.zero_impute(x)
            x_noisy = x
        encoded = self.encoder(x_noisy)
        decoded = self.decoder(encoded)
        log_hazard = self.hazard(encoded)
        return log_hazard, x, decoded


class IntermediateFusionMean(MultiModalDropout):
    def __init__(
        self,
        params: dict,
        blocks,
        p_multimodal_dropout=0.0,
        upweight=True,
        alpha=1.0,
        missing_modalities="impute",
        p_dropout=0.0,
    ) -> None:
        super().__init__(
            blocks=blocks,
            p_multimodal_dropout=p_multimodal_dropout,
            upweight=upweight,
        )
        self.hazard_input_size = params.get("fc_units", [128, 64])[-1]
        self.alpha = alpha
        self.missing_modalities = missing_modalities
        block_encoder = []

        params.update({"input_size": [len(i) for i in blocks]})
        for i in range(len(blocks)):
            params_mod = params.copy()
            params_mod.update({"input_size": params["input_size"][i]})
            params_mod["fc_dropout"] = p_dropout
            block_encoder += [Encoder(params_mod)]

        self.block_encoder = torch.nn.ModuleList(block_encoder)

        self.unimodal_log_hazards = torch.nn.ModuleList(
            [
                FCBlock(
                    params={
                        "input_size": self.hazard_input_size,
                        "fc_layers": params.get("fc_layers", 2),
                        "fc_units": params.get("hazard_fc_units", [32, 1]),
                        "fc_activation": params.get(
                            "fc_activation", ["relu", "None"]
                        ),
                        "fc_batchnorm": params.get("fc_batchnorm", True),
                        "last_layer_bias": params.get(
                            "hazard_fc_last_layer_bias", False
                        ),
                        "fc_dropout": p_dropout,
                    }
                )
            ]
            * len(blocks)
        )

        self.joint_log_hazard = FCBlock(
            params={
                "input_size": self.hazard_input_size,
                "fc_layers": params.get("fc_layers", 2),
                "fc_units": params.get("hazard_fc_units", [32, 1]),
                "fc_activation": params.get("fc_activation", ["relu", "None"]),
                "fc_batchnorm": params.get("fc_batchnorm", True),
                "last_layer_bias": params.get(
                    "hazard_fc_last_layer_bias", False
                ),
                "fc_dropout": p_dropout,
            }
        )

        self.blocks = blocks
        self.p_multimodal_dropout = p_multimodal_dropout
        self.upweight = upweight

    def forward(self, x):
        if self.training:
            if self.missing_modalities == "multimodal_dropout":
                x = self.zero_impute(x)
                x = self.multimodal_dropout(x)
            elif self.missing_modalities == "impute":
                x = self.zero_impute(x)
        else:
            x = self.zero_impute(x)

        stacked_embedding = torch.stack(
            [
                self.block_encoder[i](x[:, self.blocks[i]])
                for i in range(len(self.blocks))
            ],
            axis=1,
        )
        assert stacked_embedding.shape[1] == len(self.blocks), print(
            stacked_embedding.shape
        )
        joint_embedding = torch.mean(stacked_embedding, axis=1)
        unimodal_log_hazards = []
        for i in range(len(self.blocks)):
            unimodal_log_hazards.append(
                self.unimodal_log_hazards[i](stacked_embedding[:, i, :])
            )

        joint_log_hazard = self.joint_log_hazard(joint_embedding)
        return joint_log_hazard, unimodal_log_hazards


class IntermediateFusionPoE(MultiModalDropout):
    def __init__(
        self,
        blocks,
        params,
        beta=0.01,
        missing_modalities="poe",
        p_multimodal_dropout=0.0,
        upweight=True,
        alpha=1.0,
        p_dropout=0.0,
    ) -> None:
        super().__init__(
            blocks=blocks,
            p_multimodal_dropout=p_multimodal_dropout,
            upweight=upweight,
        )
        params.update({"fc_dropout": p_dropout})
        self.block_encoder = torch.nn.ModuleList(
            [
                FCBlock(
                    params={
                        "input_size": len(blocks[i]),
                        "fc_layers": params.get("encoder_fc_layers", 2),
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
                        "last_layer_bias": params.get(
                            "hazard_fc_last_layer_bias", "False"
                        ),
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
                "last_layer_bias": params.get(
                    "hazard_fc_last_layer_bias", "False"
                ),
                "fc_activation": params.get("fc_activation", ["relu", "None"]),
                "fc_batchnorm": params.get("fc_batchnorm", "True"),
                "fc_dropout": params.get("fc_dropout", 0.5),
            }
        )
        self.blocks = blocks
        self.params = params
        self.beta = beta
        self.missing_modalities = missing_modalities
        self.p_multimodal_dropout = p_multimodal_dropout
        self.upweight = upweight
        self.alpha = alpha
        self.p_dropout = p_dropout

    def product_of_experts(self, mask, mu, log_var):
        variance = torch.ones(mu[0].shape)
        for modality in range(len(mu)):
            variance += torch.stack(
                [mask[:, modality]] * log_var[modality].shape[1], axis=1
            ) * torch.div(1.0, torch.exp(log_var[modality]) + 1e-8)
        variance = torch.div(1.0, variance + 1e-8)
        log_variance = torch.log(variance + 1e-8)

        mu_poe = torch.zeros(variance.shape)
        for modality in range(len(mu)):
            mu_poe += (
                torch.stack(
                    [mask[:, modality]] * log_var[modality].shape[1], axis=1
                )
                * torch.div(1, torch.exp(log_var[modality]) + 1e-8)
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
        mask = torch.ones((x.shape[0], len(self.blocks)))
        if self.training:
            if self.missing_modalities == "multimodal_dropout":
                x = self.zero_impute(x)
                x = self.multimodal_dropout(x)
            elif self.missing_modalities == "impute":
                x = self.zero_impute(x)
            elif self.missing_modalities == "poe":
                mask = self.find_missing_modality_mask(x, self.blocks)
                x = self.zero_impute(x)
        else:
            if self.missing_modalities == "poe":
                mask = self.find_missing_modality_mask(x, self.blocks)
            x = self.zero_impute(x)
        mu = []
        log_var = []
        for i in range(len(self.blocks)):
            tmp = self.block_encoder[i](x[:, self.blocks[i]])
            mu.append(
                tmp[
                    :,
                    : int(
                        self.params.get("encoder_fc_units", [128, 128])[-1] / 2
                    ),
                ]
            )
            log_var.append(
                tmp[
                    :,
                    int(
                        self.params.get("encoder_fc_units", [128, 128])[-1] / 2
                    ) :,
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
