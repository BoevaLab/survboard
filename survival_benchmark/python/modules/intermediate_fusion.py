from ast import Mult
import torch
from survival_benchmark.python.modules.autoencoder import FCBlock, Encoder, Decoder
from survival_benchmark.python.utils.utils import MultiModalDropout


class IntermediateFusionMean(MultiModalDropout):
    def __init__(
        self,
        params: dict,
        blocks,
        p_multimodal_dropout=0.0,
        upweight=True,
        fusion="mean",
    ) -> None:
        super().__init__(
            blocks=blocks,
            p_multimodal_dropout=p_multimodal_dropout,
            upweight=upweight,
        )

        # self.params = params
        self.latent_dim = params.get("latent_dim", 64)
        block_encoder = []
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

        x = self.multimodal_dropout(self.zero_impute(x))

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


class IntermediateFusionPoE(MultiModalDropout):
    def __init__(
        self,
        blocks,
        missing_modalities="poe",
        p_multimodal_dropout=0.0,
        upweight=True,
        activation=torch.nn.PReLU,
        n_hidden_layers=2,
        n_first_hidden_layer=128,
        n_latent_space=64,
        p_dropout=0.5,
        batch_norm=False,
        decrease_factor_per_layer=2,
        fusion="mean",
        hazard_hidden_layer_size=[32],
        beta=1,
        beta_v=1,
        alpha=0,
    ) -> None:
        super().__init__(
            blocks=blocks,
            p_multimodal_dropout=p_multimodal_dropout,
            upweight=upweight,
        )
        self.block_encoder = torch.nn.ModuleList(
            [
                Encoder(
                    params={
                        "input_size": len(blocks[i]),
                        "fc_layers": 2,
                        "fc_units": [128, 128],
                        "fc_activation": "relu",
                        "fc_batchnorm": True,
                        "fc_dropout": p_dropout,
                    }
                )
                for i in range(len(blocks))
            ]
        )
        self.unimodal_log_hazards = torch.nn.ModuleList(
            [
                FCBlock(
                    params={
                        "input_size": n_latent_space,
                        "fc_layers": 2,
                        "fc_units": [64, 1],
                        "fc_activation": fc_activation,
                        "fc_batchnorm": fc_batchnorm,
                        "last_layer_bias": False,
                    }
                )
            ]
            * len(blocks)
        )
        self.joint_log_hazard = FCBlock(
            params={
                "input_size": n_latent_space,
                "fc_layers": 2,
                "fc_units": [64, 1],
                "fc_activation": fc_activation,
                "fc_batchnorm": fc_batchnorm,
                "last_layer_bias": False,
            }
        )
        self.blocks = blocks
        self.missing_modalities = missing_modalities
        self.p_multimodal_dropout = p_multimodal_dropout
        self.upweight = upweight
        self.activation = activation
        self.n_hidden_layers = n_hidden_layers
        self.n_first_hidden_layer = n_first_hidden_layer
        self.n_latent_space = n_latent_space
        self.p_dropout = p_dropout
        self.batch_norm = batch_norm
        self.decrease_factor_per_layer = decrease_factor_per_layer
        self.fusion = fusion
        self.beta = beta
        self.beta_v = beta_v
        self.alpha = alpha

    def product_of_experts(self, mask, mu, log_var):
        variance = torch.eye(mu.shape)
        for modality in range(len(mu)):
            variance += mask[:, modality] * torch.div(1.0, torch.exp(log_var[modality]))
        variance = torch.div(1.0, variance)
        log_variance = torch.log(variance)

        mu = torch.zeros(mu.shape)
        for modality in range(len(mu)):
            mu += mask[:, modality] * torch.div(1, torch.exp(log_var[modality])) * mu[modality]
        mu = variance * mu
        return mu, log_variance

    def find_missing_modality_mask(self, x, blocks):
        mask = torch.zeros((x.shape[0], len(blocks)))
        for modality in blocks:
            mask[:, modality] = (torch.sum(torch.isnan(x[:, blocks[modality]]), axis=1) != len(blocks[modality])).long()
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
            mu.append(tmp[0])
            log_var.append(tmp[1])

        joint_mu, joint_log_var = self.product_of_experts(mask, mu, log_var)
        joint_posterior_distribution = torch.distributions.normal.Normal(joint_mu, torch.sqrt(torch.exp(joint_log_var)))
        joint_posterior = joint_posterior_distribution.rsample(joint_mu.shape)
        joint_log_hazard = self.joint_log_hazard(joint_posterior)
        unimodal_posterior_distributions = []
        unimodal_posteriors = []
        unimodal_log_hazard = []
        for ix in range(len(mu)):
            unimodal_posterior_distributions[ix] = torch.distributions.normal.Normal(
                mu[ix], torch.sqrt(torch.exp(log_var[ix]))
            )
            unimodal_posteriors[ix] = unimodal_posterior_distributions[ix].rsample(joint_mu.shape)
            unimodal_log_hazard[ix] = self.unimodal_log_hazard[ix](unimodal_posteriors[ix])

        return (
            joint_log_hazard,
            joint_posterior,
            joint_posterior_distribution,
            unimodal_log_hazard,
            unimodal_posteriors,
            unimodal_posterior_distributions,
        )


class poe_criterion(torch.nn.Module):
    def forward(
        self,
        target,
        predicted,
        alpha,
        beta,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        std_normal = torch.distributions.normal.Normal(0, 1).to(device)
        time, event = inverse_transform_target(target)
        cox_joint = negative_partial_log_likelihood(predicted[0], time, event).to(device)
        cox_modality = [
            negative_partial_log_likelihood(log_hazard, time, event).to(device) for log_hazard in predicted[3]
        ]
        joint_kl = torch.distributions.kl.kl_divergence(predicted[2], std_normal).to(device)

        modality_kl = [
            torch.distributions.kl.kl_divergence(posterior, std_normal).to(device) for posterior in predicted[5]
        ]
        return (
            cox_joint
            + beta * joint_kl
            + alpha * (torch.sum(torch.cat(cox_modality)) + beta * torch.sum(torch.cat(modality_kl)))
        )
