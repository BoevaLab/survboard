import torch

from survival_benchmark.python.autoencoder import Encoder, FCBlock
from survival_benchmark.python.utils.utils import MultiModalDropout


class NaiveNeural(torch.nn.Module):
    def __init__(
        self,
        params,
        blocks,
        missing_modalities,
        p_multimodal_dropout,
        p_dropout=0.0,
    ) -> None:
        super().__init__()
        self.input_size = sum([len(i) for i in blocks])
        self.latent_dim = params.get("latent_dim")
        self.hidden_units = params.get("fc_units")
        self.params = params
        self.params["fc_dropout"] = p_dropout
        params_encoder = self.params.copy()
        params_encoder.update({"input_size": self.input_size})
        self.encoder = Encoder(params_encoder)
        fc_params = {
            "input_size": self.input_size,
            "latent_dim": 1,
            "last_layer_bias": self.params.get("hazard_fc_last_layer_bias"),
            "fc_layers": 3,
            "fc_units": [128, 64, 1],
            "fc_dropout": self.params.get("fc_dropout"),
        }
        self.log_hazard = FCBlock(fc_params)
        self.hazard = FCBlock(fc_params)

    def forward(self, x):
        log_hazard = self.log_hazard(x)
        return log_hazard


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
