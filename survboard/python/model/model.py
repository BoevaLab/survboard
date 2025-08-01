import torch

from survboard.python.utils.factories import FUSION_FACTORY
from survboard.python.utils.misc_utils import (
    calculate_log_hazard_input_size,
    multimodal_dropout,
)


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


class HazardRegressionSurvivalNet(torch.nn.Module):
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
            next_size = hidden_layer_size
            hazard.append(torch.nn.Linear(current_size, next_size))
            hazard.append(activation())
            hazard.append(torch.nn.Dropout(p_dropout))
            current_size = next_size
        hazard.append(torch.nn.Linear(current_size, output_size, bias=with_bias))
        self.hazard = torch.nn.Sequential(*hazard)

    def forward(self, x):
        return torch.clamp(self.hazard(x), -75, 75)


class HazardRegressionGDP(torch.nn.Module):
    def __init__(
        self,
        input_dimension,
        output_size=1,
        hidden_layer_size=100,
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
            next_size = hidden_layer_size // max(int(layer + 1), 1)
            hazard.append(torch.nn.Linear(current_size, next_size))
            hazard.append(activation())
            hazard.append(torch.nn.Dropout(p_dropout))
            current_size = next_size
        hazard.append(torch.nn.Linear(current_size, output_size, bias=with_bias))
        self.hazard = torch.nn.Sequential(*hazard)

    def forward(self, x):
        return torch.clamp(self.hazard(x), -75, 75)


class EncoderSalmon(torch.nn.Module):
    def __init__(
        self,
        input_dimension,
        hidden_layer_size,
        activation=torch.nn.Sigmoid,
        p_dropout=0.5,
    ):
        super().__init__()
        hazard = []
        current_size = input_dimension

        next_size = hidden_layer_size
        hazard.append(torch.nn.Linear(current_size, next_size))
        hazard.append(activation())
        hazard.append(torch.nn.Dropout(p_dropout))
        self.hazard = torch.nn.Sequential(*hazard)

    def forward(self, x):
        return torch.clamp(self.hazard(x), -75, 75)


class BaseFusionModel(torch.nn.Module):
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

    def forward(self, x):
        x = self.zero_impute(x)
        if self.training and self.p_multimodal_dropout > 0.0:
            x = self.multimodal_dropout(x)
        fused = self.fusion(x)
        if "late" in self.fusion_method:
            return fused
        else:
            if len(fused) > 1 and isinstance(fused, tuple):
                if self.fusion_method == "intermediate_ae":
                    return (self.log_hazard(fused[0]),) + fused[1:] + (x,)
                else:
                    return (self.log_hazard(fused[0]),) + fused[1:]
            else:
                return self.log_hazard(fused)


class CoxPHNeural(BaseFusionModel):
    def __init__(
        self,
        fusion_method,
        blocks,
        activation=torch.nn.PReLU,
        p_dropout=0.0,
        log_hazard_hidden_layer_size=64,
        log_hazard_hidden_layers=1,
        modality_hidden_layer_size=128,
        modality_hidden_layers=1,
        p_multimodal_dropout=0.0,
        upweight=True,
    ) -> None:
        super().__init__()
        self.fusion_method = fusion_method
        self.blocks = blocks
        self.activation = activation
        self.p_dropout = p_dropout
        self.log_hazard_hidden_layer_size = log_hazard_hidden_layer_size
        self.log_hazard_hidden_layers = log_hazard_hidden_layers
        self.modality_hidden_layer_size = modality_hidden_layer_size
        self.modality_hidden_layers = modality_hidden_layers
        if "late" in fusion_method:
            self.fusion = FUSION_FACTORY[fusion_method](
                output_size=1,
                blocks=blocks,
                activation=activation,
                p_dropout=p_dropout,
                log_hazard_hidden_layer_size=log_hazard_hidden_layer_size,
                log_hazard_hidden_layers=log_hazard_hidden_layers,
                modality_hidden_layer_size=modality_hidden_layer_size,
                modality_hidden_layers=modality_hidden_layers,
            )
        else:
            self.fusion = FUSION_FACTORY[fusion_method](
                blocks=blocks,
                activation=activation,
                p_dropout=p_dropout,
                log_hazard_hidden_layer_size=log_hazard_hidden_layer_size,
                log_hazard_hidden_layers=log_hazard_hidden_layers,
                modality_hidden_layer_size=modality_hidden_layer_size,
                modality_hidden_layers=modality_hidden_layers,
            )
        self.p_multimodal_dropout = p_multimodal_dropout
        self.upweight = upweight
        if "late" not in fusion_method:
            self.log_hazard = HazardRegression(
                input_dimension=calculate_log_hazard_input_size(
                    fusion_method=fusion_method,
                    blocks=blocks,
                    modality_dimension=modality_hidden_layer_size,
                ),
                output_size=1,
                hidden_layer_size=log_hazard_hidden_layer_size
                // (1 + int(fusion_method == "early_ae")),
                activation=activation,
                hidden_layers=log_hazard_hidden_layers
                - int(fusion_method == "early_ae"),
                p_dropout=p_dropout,
            )


class EHNeural(BaseFusionModel):
    def __init__(
        self,
        fusion_method,
        blocks,
        activation=torch.nn.PReLU,
        p_dropout=0.0,
        log_hazard_hidden_layer_size=64,
        log_hazard_hidden_layers=1,
        modality_hidden_layer_size=128,
        modality_hidden_layers=1,
        p_multimodal_dropout=0.0,
        upweight=True,
    ) -> None:
        super().__init__()
        self.fusion_method = fusion_method
        self.blocks = blocks
        self.activation = activation
        self.p_dropout = p_dropout
        self.log_hazard_hidden_layer_size = log_hazard_hidden_layer_size
        self.log_hazard_hidden_layers = log_hazard_hidden_layers
        self.modality_hidden_layer_size = modality_hidden_layer_size
        self.modality_hidden_layers = modality_hidden_layers
        if "late" in fusion_method:
            self.fusion = FUSION_FACTORY[fusion_method](
                output_size=2,
                blocks=blocks,
                activation=activation,
                p_dropout=p_dropout,
                log_hazard_hidden_layer_size=log_hazard_hidden_layer_size,
                log_hazard_hidden_layers=log_hazard_hidden_layers,
                modality_hidden_layer_size=modality_hidden_layer_size,
                modality_hidden_layers=modality_hidden_layers,
            )
        else:
            self.fusion = FUSION_FACTORY[fusion_method](
                blocks=blocks,
                activation=activation,
                p_dropout=p_dropout,
                log_hazard_hidden_layer_size=log_hazard_hidden_layer_size,
                log_hazard_hidden_layers=log_hazard_hidden_layers,
                modality_hidden_layer_size=modality_hidden_layer_size,
                modality_hidden_layers=modality_hidden_layers,
            )
        self.p_multimodal_dropout = p_multimodal_dropout
        self.upweight = upweight
        if "late" not in fusion_method:
            self.log_hazard = HazardRegression(
                input_dimension=calculate_log_hazard_input_size(
                    fusion_method=fusion_method,
                    blocks=blocks,
                    modality_dimension=modality_hidden_layer_size,
                ),
                output_size=2,
                hidden_layer_size=log_hazard_hidden_layer_size
                // (1 + int(fusion_method == "early_ae")),
                activation=activation,
                hidden_layers=log_hazard_hidden_layers
                - int(fusion_method == "early_ae"),
                p_dropout=p_dropout,
            )


class DiscreteNeural(BaseFusionModel):
    def __init__(
        self,
        fusion_method,
        blocks,
        n_output_points=30,
        activation=torch.nn.PReLU,
        p_dropout=0.0,
        log_hazard_hidden_layer_size=64,
        log_hazard_hidden_layers=1,
        modality_hidden_layer_size=128,
        modality_hidden_layers=1,
        p_multimodal_dropout=0.0,
        upweight=True,
    ) -> None:
        super().__init__()
        self.fusion_method = fusion_method
        self.blocks = blocks
        self.activation = activation
        self.p_dropout = p_dropout
        self.log_hazard_hidden_layer_size = log_hazard_hidden_layer_size
        self.log_hazard_hidden_layers = log_hazard_hidden_layers
        self.modality_hidden_layer_size = modality_hidden_layer_size
        self.modality_hidden_layers = modality_hidden_layers
        self.n_output_points = n_output_points
        if "late" in fusion_method:
            self.fusion = FUSION_FACTORY[fusion_method](
                output_size=n_output_points,
                blocks=blocks,
                activation=activation,
                p_dropout=p_dropout,
                log_hazard_hidden_layer_size=log_hazard_hidden_layer_size,
                log_hazard_hidden_layers=log_hazard_hidden_layers,
                modality_hidden_layer_size=modality_hidden_layer_size,
                modality_hidden_layers=modality_hidden_layers,
                with_bias=True,
            )
        else:
            self.fusion = FUSION_FACTORY[fusion_method](
                blocks=blocks,
                activation=activation,
                p_dropout=p_dropout,
                log_hazard_hidden_layer_size=log_hazard_hidden_layer_size,
                log_hazard_hidden_layers=log_hazard_hidden_layers,
                modality_hidden_layer_size=modality_hidden_layer_size,
                modality_hidden_layers=modality_hidden_layers,
            )
        self.p_multimodal_dropout = p_multimodal_dropout
        self.upweight = upweight
        if "late" not in fusion_method:
            self.log_hazard = HazardRegression(
                input_dimension=calculate_log_hazard_input_size(
                    fusion_method=fusion_method,
                    blocks=blocks,
                    modality_dimension=modality_hidden_layer_size,
                ),
                output_size=n_output_points,
                hidden_layer_size=log_hazard_hidden_layer_size
                // (1 + int(fusion_method == "early_ae")),
                activation=activation,
                hidden_layers=log_hazard_hidden_layers
                - int(fusion_method == "early_ae"),
                p_dropout=p_dropout,
                with_bias=True,
            )


class DiscreteNeuralSGL(DiscreteNeural):
    def __init__(
        self,
        fusion_method,
        blocks,
        alpha=0,
        lamb=0,
        n_output_points=30,
        activation=torch.nn.PReLU,
        p_dropout=0.0,
        log_hazard_hidden_layer_size=64,
        log_hazard_hidden_layers=1,
        modality_hidden_layer_size=128,
        modality_hidden_layers=1,
        p_multimodal_dropout=0.0,
        upweight=True,
    ) -> None:
        super().__init__(
            fusion_method=fusion_method,
            blocks=blocks,
            n_output_points=n_output_points,
            activation=activation,
            p_dropout=p_dropout,
            log_hazard_hidden_layer_size=log_hazard_hidden_layer_size,
            log_hazard_hidden_layers=log_hazard_hidden_layers,
            modality_hidden_layer_size=modality_hidden_layer_size,
            modality_hidden_layers=modality_hidden_layers,
            p_multimodal_dropout=p_multimodal_dropout,
            upweight=upweight,
        )
        self.alpha = alpha
        self.lamb = lamb


class CoxPHNeuralSGL(CoxPHNeural):
    def __init__(
        self,
        fusion_method,
        blocks,
        alpha=0,
        lamb=0,
        activation=torch.nn.PReLU,
        p_dropout=0.0,
        log_hazard_hidden_layer_size=64,
        log_hazard_hidden_layers=1,
        modality_hidden_layer_size=128,
        modality_hidden_layers=1,
        p_multimodal_dropout=0.0,
        upweight=True,
    ) -> None:
        super().__init__(
            fusion_method=fusion_method,
            blocks=blocks,
            activation=activation,
            p_dropout=p_dropout,
            log_hazard_hidden_layer_size=log_hazard_hidden_layer_size,
            log_hazard_hidden_layers=log_hazard_hidden_layers,
            modality_hidden_layer_size=modality_hidden_layer_size,
            modality_hidden_layers=modality_hidden_layers,
            p_multimodal_dropout=p_multimodal_dropout,
            upweight=upweight,
        )
        self.alpha = alpha
        self.lamb = lamb


class EHNeuralSGL(EHNeural):
    def __init__(
        self,
        fusion_method,
        blocks,
        alpha=0,
        lamb=0,
        activation=torch.nn.PReLU,
        p_dropout=0.0,
        log_hazard_hidden_layer_size=64,
        log_hazard_hidden_layers=1,
        modality_hidden_layer_size=128,
        modality_hidden_layers=1,
        p_multimodal_dropout=0.0,
        upweight=True,
    ) -> None:
        super().__init__(
            fusion_method=fusion_method,
            blocks=blocks,
            activation=activation,
            p_dropout=p_dropout,
            log_hazard_hidden_layer_size=log_hazard_hidden_layer_size,
            log_hazard_hidden_layers=log_hazard_hidden_layers,
            modality_hidden_layer_size=modality_hidden_layer_size,
            modality_hidden_layers=modality_hidden_layers,
            p_multimodal_dropout=p_multimodal_dropout,
            upweight=upweight,
        )
        self.alpha = alpha
        self.lamb = lamb


class SurvivalNet(BaseFusionModel):
    def __init__(
        self,
        blocks,
        activation,
        fusion_method="early",
        p_dropout=0.0,
        log_hazard_hidden_layer_size=64,
        log_hazard_hidden_layers=0,
        modality_hidden_layer_size=128,
        modality_hidden_layers=1,
        p_multimodal_dropout=0.0,
        upweight=True,
    ) -> None:
        super().__init__()
        self.fusion_method = fusion_method
        self.blocks = blocks
        self.activation = activation
        self.p_dropout = p_dropout
        self.log_hazard_hidden_layer_size = log_hazard_hidden_layer_size
        self.log_hazard_hidden_layers = log_hazard_hidden_layers
        self.modality_hidden_layer_size = modality_hidden_layer_size
        self.modality_hidden_layers = modality_hidden_layers
        if "late" in fusion_method:
            self.fusion = FUSION_FACTORY[fusion_method](
                output_size=1,
                blocks=blocks,
                activation=activation,
                p_dropout=p_dropout,
                log_hazard_hidden_layer_size=log_hazard_hidden_layer_size,
                log_hazard_hidden_layers=log_hazard_hidden_layers,
                modality_hidden_layer_size=modality_hidden_layer_size,
                modality_hidden_layers=modality_hidden_layers,
            )
        else:
            self.fusion = FUSION_FACTORY[fusion_method](
                blocks=blocks,
                activation=activation,
                p_dropout=p_dropout,
                log_hazard_hidden_layer_size=log_hazard_hidden_layer_size,
                log_hazard_hidden_layers=log_hazard_hidden_layers,
                modality_hidden_layer_size=modality_hidden_layer_size,
                modality_hidden_layers=modality_hidden_layers,
            )
        self.p_multimodal_dropout = p_multimodal_dropout
        self.upweight = upweight
        if "late" not in fusion_method:
            self.log_hazard = HazardRegressionSurvivalNet(
                input_dimension=calculate_log_hazard_input_size(
                    fusion_method="early",
                    blocks=blocks,
                    modality_dimension=modality_hidden_layer_size,
                ),
                output_size=1,
                hidden_layer_size=log_hazard_hidden_layer_size,
                activation=activation,
                hidden_layers=log_hazard_hidden_layers,
                p_dropout=p_dropout,
                with_batchnorm=False,
            )


class GDP(BaseFusionModel):
    def __init__(
        self,
        blocks,
        activation,
        alpha=0,
        lamb=0,
        fusion_method="early",
        p_dropout=0.0,
        log_hazard_hidden_layer_size=200,
        log_hazard_hidden_layers=2,
        modality_hidden_layer_size=128,
        modality_hidden_layers=1,
        p_multimodal_dropout=0.0,
        upweight=True,
    ) -> None:
        super().__init__()
        self.fusion_method = fusion_method
        self.blocks = blocks
        self.activation = activation
        self.p_dropout = p_dropout
        self.log_hazard_hidden_layer_size = log_hazard_hidden_layer_size
        self.log_hazard_hidden_layers = log_hazard_hidden_layers
        self.modality_hidden_layer_size = modality_hidden_layer_size
        self.modality_hidden_layers = modality_hidden_layers

        self.fusion = FUSION_FACTORY["early"](
            blocks=blocks,
            activation=activation,
            p_dropout=p_dropout,
            log_hazard_hidden_layer_size=log_hazard_hidden_layer_size,
            log_hazard_hidden_layers=log_hazard_hidden_layers,
            modality_hidden_layer_size=modality_hidden_layer_size,
            modality_hidden_layers=modality_hidden_layers,
        )
        self.p_multimodal_dropout = p_multimodal_dropout
        self.upweight = upweight

        self.log_hazard = HazardRegressionGDP(
            input_dimension=calculate_log_hazard_input_size(
                fusion_method="early",
                blocks=blocks,
                modality_dimension=modality_hidden_layer_size,
            ),
            output_size=1,
            hidden_layer_size=log_hazard_hidden_layer_size,
            activation=activation,
            hidden_layers=log_hazard_hidden_layers,
            p_dropout=p_dropout,
            with_batchnorm=False,
        )
        self.alpha = alpha
        self.lamb = lamb


class Salmon(torch.nn.Module):
    def __init__(
        self,
        blocks,
        modality_hidden_layer_sizes,
        activation=torch.nn.Sigmoid,
        lamb=0,
        p_dropout=0.0,
        upweight=True,
    ) -> None:
        super().__init__()
        self.blocks = blocks
        self.activation = activation
        self.p_dropout = p_dropout
        self.modality_hidden_layer_sizes = modality_hidden_layer_sizes
        self.modality_encoders = torch.nn.ModuleList(
            [
                EncoderSalmon(
                    input_dimension=len(blocks[i]),
                    hidden_layer_size=modality_hidden_layer_sizes[i],
                    activation=activation,
                    p_dropout=p_dropout,
                )
                for i in range(len(blocks[:-1]))
            ]
        )
        self.log_hazard = HazardRegression(
            input_dimension=(sum(modality_hidden_layer_sizes) + len(blocks[-1])),
            output_size=1,
            hidden_layers=0,
            with_bias=False,
        )
        self.lamb = lamb

    def forward(self, x):
        modality_encodings = []
        for ix, modality in enumerate(self.blocks[:-1]):
            modality_encodings.append(self.modality_encoders[ix](x[:, modality]))
        modality_encodings.append(x[:, self.blocks[-1]])
        x_concatenated = torch.concat(modality_encodings, axis=1)
        log_hazard = self.log_hazard(x_concatenated)
        return log_hazard


SKORCH_MODULE_FACTORY = {
    "cox": CoxPHNeural,
    "eh": EHNeural,
    "discrete_time": DiscreteNeural,
    "cox_sgl": CoxPHNeuralSGL,
    "eh_sgl": EHNeuralSGL,
    "discrete_time_sgl": DiscreteNeuralSGL,
    "survival_net": SurvivalNet,
    "gdp": GDP,
    "salmon": Salmon,
}
