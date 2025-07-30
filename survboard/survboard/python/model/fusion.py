import torch
import torch.nn.functional as F

# All fusion code adapted from: https://github.com/BoevaLab/Multi-omics-noise-resistance


def multimodal_dropout(x, p_multimodal_dropout, blocks, upweight=True):
    for block in blocks:
        if not torch.all(x[:, block] == 0):
            msk = torch.where((torch.rand(x.shape[0]) <= p_multimodal_dropout).long())[
                0
            ]
            x[:, torch.tensor(block)][msk, :] = torch.zeros(
                x[:, torch.tensor(block)][msk, :].shape
            )

    if upweight:
        x = x / (1 - p_multimodal_dropout)
    return x


class MultiModalDropout(torch.nn.Module):
    def __init__(self, blocks, p_multimodal_dropout=0.0, upweight=True) -> None:
        super().__init__()
        self.blocks = blocks
        self.p_multimodal_dropout = p_multimodal_dropout
        self.upweight = upweight

    def zero_impute(self, x):
        return torch.nan_to_num(x, nan=0.0)

    def multimodal_dropout(self, x):
        if self.p_multimodal_dropout > 0 and self.training:
            x = multimodal_dropout(
                x=x,
                blocks=self.blocks,
                p_multimodal_dropout=self.p_multimodal_dropout,
                upweight=self.upweight,
            )
        return x


class HazardRegression(torch.nn.Module):
    def __init__(
        self,
        input_dimension,
        output_size=1,
        hidden_layer_size=32,
        activation=torch.nn.ReLU,
        hidden_layers=1,
        p_dropout=0.5,
        with_bias=False,
        with_batchnorm=True,
    ):
        super().__init__()
        hazard = []
        current_size = input_dimension
        for layer in range(hidden_layers):
            next_size = hidden_layer_size // (1 + layer)
            hazard.append(torch.nn.Linear(current_size, next_size))
            hazard.append(activation())
            hazard.append(torch.nn.Dropout(p_dropout))
            if with_batchnorm:
                hazard.append(torch.nn.BatchNorm1d(next_size))
            current_size = next_size
        hazard.append(torch.nn.Linear(current_size, output_size, bias=with_bias))
        self.hazard = torch.nn.Sequential(*hazard)

    def forward(self, x):
        return torch.clamp(self.hazard(x), -75, 75)


class Encoder(torch.nn.Module):
    def __init__(
        self,
        input_dimension,
        hidden_layer_size=64,
        activation=torch.nn.ReLU,
        hidden_layers=1,
        p_dropout=0.5,
        with_batchnorm=True,
    ):
        super().__init__()
        encoder = []
        current_size = input_dimension
        next_size = hidden_layer_size
        for i in range(hidden_layers):
            encoder.append(torch.nn.Linear(current_size, next_size))
            encoder.append(activation())
            encoder.append(torch.nn.Dropout(p_dropout))
            if with_batchnorm:
                encoder.append(torch.nn.BatchNorm1d(next_size))
            current_size = next_size
        self.encode = torch.nn.Sequential(*encoder)
        self.input_dimension = input_dimension
        self.hidden_layer_size = hidden_layer_size
        self.activation = activation
        self.hidden_layers = hidden_layers

    def forward(self, x):
        return self.encode(x)


class Fusion(torch.nn.Module):
    def __init__(
        self,
        blocks,
        activation,
        p_dropout,
        log_hazard_hidden_layer_size,
        log_hazard_hidden_layers,
        modality_hidden_layer_size,
        modality_hidden_layers,
    ) -> None:
        super().__init__()
        self.blocks = blocks
        self.activation = activation
        self.p_dropout = p_dropout
        self.log_hazard_hidden_layer_size = log_hazard_hidden_layer_size
        self.log_hazard_hidden_layers = log_hazard_hidden_layers
        self.modality_hidden_layer_size = modality_hidden_layer_size
        self.modality_hidden_layers = modality_hidden_layers

    def forward(self, x):
        raise NotImplementedError


class EarlyFusion(Fusion):
    def __init__(
        self,
        blocks,
        activation,
        p_dropout,
        log_hazard_hidden_layer_size,
        log_hazard_hidden_layers,
        modality_hidden_layer_size,
        modality_hidden_layers,
    ) -> None:
        super().__init__(
            blocks,
            activation,
            p_dropout,
            log_hazard_hidden_layer_size,
            log_hazard_hidden_layers,
            modality_hidden_layer_size,
            modality_hidden_layers,
        )

    def forward(self, x):
        return x


class LateFusion(Fusion):
    def __init__(
        self,
        output_size,
        blocks,
        activation,
        p_dropout,
        log_hazard_hidden_layer_size,
        log_hazard_hidden_layers,
        modality_hidden_layer_size,
        modality_hidden_layers,
        with_bias=False,
    ) -> None:
        super().__init__(
            blocks,
            activation,
            p_dropout,
            log_hazard_hidden_layer_size,
            log_hazard_hidden_layers,
            modality_hidden_layer_size,
            modality_hidden_layers,
        )
        self.output_size = output_size
        self.modality_specific_log_hazard_ratio = torch.nn.ModuleList(
            [
                HazardRegression(
                    input_dimension=len(blocks[i]),
                    output_size=output_size,
                    hidden_layer_size=log_hazard_hidden_layer_size,
                    activation=activation,
                    hidden_layers=log_hazard_hidden_layers,
                    p_dropout=p_dropout,
                    with_bias=with_bias,
                )
                for i in range(len(blocks))
            ]
        )

    def get_weights(self, x):
        raise NotImplementedError

    def forward(self, x):
        return torch.squeeze(
            torch.matmul(
                torch.einsum(
                    "ijk -> jki",
                    (
                        torch.stack(
                            [
                                self.modality_specific_log_hazard_ratio[i](
                                    x[:, self.blocks[i]]
                                )
                                for i in range(len(self.blocks))
                            ],
                            dim=0,
                        )
                    ),
                ),
                (self.get_weights(x)),
            )
        )


class LateFusionMean(LateFusion):
    def get_weights(self, x):
        return torch.full_like(torch.ones((len(self.blocks), 1)), 1 / len(self.blocks))


class IntermediateFusion(Fusion):
    def __init__(
        self,
        blocks,
        activation,
        p_dropout,
        log_hazard_hidden_layer_size,
        log_hazard_hidden_layers,
        modality_hidden_layer_size,
        modality_hidden_layers,
    ) -> None:
        super().__init__(
            blocks,
            activation,
            p_dropout,
            log_hazard_hidden_layer_size,
            log_hazard_hidden_layers,
            modality_hidden_layer_size,
            modality_hidden_layers,
        )
        self.modality_encoders = torch.nn.ModuleList(
            [
                Encoder(
                    input_dimension=len(blocks[i]),
                    hidden_layer_size=modality_hidden_layer_size,
                    activation=activation,
                    hidden_layers=modality_hidden_layers,
                    p_dropout=p_dropout,
                )
                for i in range(len(blocks))
            ]
        )

    def fusion(self, x):
        raise NotImplementedError

    def forward(self, x):
        modality_encodings = []
        for ix, modality in enumerate(self.blocks):
            modality_encodings.append(self.modality_encoders[ix](x[:, modality]))

        return self.fusion(modality_encodings)


class IntermediateFusionMean(IntermediateFusion):
    def fusion(self, x):
        return torch.mean(torch.stack(x), axis=0)


class IntermediateFusionMax(IntermediateFusion):
    def fusion(self, x):
        # `torch.max` returns a tuple of (max, max_indices),
        # so we select the maximum as the first element.
        return torch.max(torch.stack(x), axis=0)[0]


class IntermediateFusionConcat(IntermediateFusion):
    def fusion(self, x):
        return torch.concat(x, axis=1)


# Adapted from: https://github.com/ZhangqiJiang07/MultimodalSurvivalPrediction/blob/main/models/sub_models/attention.py
class IntermediateFusionAttention(IntermediateFusion):
    def __init__(
        self,
        blocks,
        activation,
        p_dropout,
        log_hazard_hidden_layer_size,
        log_hazard_hidden_layers,
        modality_hidden_layer_size,
        modality_hidden_layers,
    ) -> None:
        super().__init__(
            blocks,
            activation,
            p_dropout,
            log_hazard_hidden_layer_size,
            log_hazard_hidden_layers,
            modality_hidden_layer_size,
            modality_hidden_layers,
        )
        self.attention_weights = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    modality_hidden_layer_size, modality_hidden_layer_size, bias=False
                )
                for i in range(len(blocks))
            ]
        )

    def fusion(self, x):
        attention_weight = tuple()
        multimodal_features = tuple()
        for modality in range(len(x)):
            attention_weight += (
                torch.tanh(self.attention_weights[modality](x[modality])),
            )
            multimodal_features += (x[modality],)
        attention_matrix = F.softmax(torch.stack(attention_weight), dim=0)
        fused_vec = torch.sum(
            torch.stack(multimodal_features) * attention_matrix, dim=0
        )
        return fused_vec
