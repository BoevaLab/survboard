import torch
import torch.nn as nn
from survival_benchmark.python.utils.hyperparameters import ACTIVATION_FN_FACTORY

from survival_benchmark.python.utils.utils import MultiModalDropout
from survival_benchmark.python.utils.utils import negative_partial_log_likelihood


class FCBlock(nn.Module):
    """Generalisable DNN module to allow for flexibility in architecture."""

    def __init__(self, params: dict) -> None:
        """Constructor.

        Args:
            params (dict): DNN parameter dictionary with the following keys:
                input_size (int): Input tensor dimensions.
                fc_layers (int): Number of fully connected layers to add.
                fc_units (List[(int)]): List of hidden units for each layer.
                fc_activation (str): Activation function to apply after each
                    fully connected layer. See utils/hyperparameter.py
                    for options.
                fc_batchnorm (bool): Whether to include a batchnorm layer.
                fc_dropout (float): Probability of dropout applied after eacch layer.
                last_layer_bias (bool): True if bias should be applied in the last layer, False otherwise. Default True.

        """
        super(FCBlock, self).__init__()

        self.input_size = params.get("input_size", 256)
        self.latent_dim = params.get("latent_dim", 64)

        self.hidden_size = params.get("fc_units", [128, 64])
        self.layers = params.get("fc_layers", 2)
        self.scaling_factor = params.get("scaling_factor", 0.5)

        self.activation = params.get("fc_activation", ["relu", "None"])
        self.dropout = params.get("fc_dropout", 0.5)
        self.batchnorm = eval(params.get("fc_batchnorm", "True"))
        self.bias_last = eval(params.get("last_layer_bias", "True"))
        bias = [True] * (self.layers - 1) + [self.bias_last]

        if len(self.hidden_size) != self.layers and self.scaling_factor is not None:
            hidden_size_generated = [self.input_size]
            # factor = (self.reduction_factor - 1) / self.reduction_factor
            for layer in range(self.layers):
                try:
                    hidden_size_generated.append(self.hidden_size[layer])
                except IndexError:
                    if layer == self.layers - 1:
                        hidden_size_generated.append(self.latent_dim)
                    else:
                        hidden_size_generated.append(int(hidden_size_generated[-1] * self.scaling_factor))

            self.hidden_size = hidden_size_generated[1:]

        if len(self.activation) != self.layers:
            if len(self.activation) == 2:
                first, last = self.activation
                self.activation = [first] * (self.layers - 1) + [last]

            elif len(self.activation) == 1:
                self.activation = self.activation * self.layers

            else:
                raise ValueError

        modules = []
        self.hidden_units = [self.input_size] + self.hidden_size
        for layer in range(self.layers):
            modules.append(
                nn.Linear(int(self.hidden_units[layer]), int(self.hidden_units[layer + 1]), bias=bias[layer])
            )
            if self.activation[layer] != "None":
                modules.append(ACTIVATION_FN_FACTORY[self.activation[layer]])
            if self.dropout > 0:
                if layer < self.layers - 1:
                    modules.append(nn.Dropout(self.dropout))
            if self.batchnorm:
                if layer < self.layers - 1:
                    modules.append(nn.BatchNorm1d(self.hidden_units[layer + 1]))
        self.model = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passes input through a feed forward neural network.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size,*,input_size]

        Returns:
            torch.Tensor: Output tensor of shape [batch_size,*, hidden_sizes[-1]].
        """

        return self.model(x)


class Encoder(nn.Module):
    def __init__(
        self, params: dict, device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ) -> None:
        super().__init__()
        self.device = device
        self.enc_params = params
        self.encoder = FCBlock(self.enc_params)

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(
        self, params: dict, device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ) -> None:
        super().__init__()
        self.device = device
        self.dec_params = params
        self.decoder = FCBlock(self.dec_params)

    def forward(self, x):
        return self.decoder(x)


class DAE(MultiModalDropout):
    def __init__(
        self,
        params,
        blocks,
        missing_modalities="impute",
        p_multimodal_dropout=0.0,
        noise_factor=0,
        alpha=0.1,
        upweight=True,
    ) -> None:
        super().__init__(blocks=blocks, p_multimodal_dropout=p_multimodal_dropout, upweight=upweight)

        self.alpha = alpha
        self.noise_factor = noise_factor
        self.missing_modalities = missing_modalities

        self.input_size = params.get("input_size")
        self.latent_dim = params.get("latent_dim")
        self.hidden_units = params.get("fc_units")
        self.params = params
        self.encoder = Encoder(params)

        self.params.update(
            {
                "input_size": self.latent_dim,
                "latent_dim": self.input_size,
                "fc_units": self.hidden_units[-2::-1],
                "scaling_factor": 2,
            }
        )

        self.decoder = Decoder(params)

        fc_params = {
            "input_size": self.latent_dim,
            "latent_dim": 1,
            "last_layer_bias": "False",
            "n_layers": 2,
            "fc_units": [int(self.latent_dim / 2), 1],
        }

        self.hazard = FCBlock(fc_params)

    def forward(self, x):
        x = self.zero_impute(x)
        x_dropout = x
        if self.missing_modalities == "multimodal_dropout":
            x_dropout = self.multimodal_dropout(x)
        x_noisy = x_dropout + (self.noise_factor * torch.normal(mean=0.0, std=1, size=x_dropout.shape))
        encoded = self.encoder(x_noisy)
        decoded = self.decoder(encoded)
        log_hazard = self.hazard(encoded)
        return log_hazard, x, decoded


class dae_criterion(nn.Module):
    def forward(self, predicted, target, alpha):
        time, event = target[:, 0], target[:, 1]
        cox_loss = negative_partial_log_likelihood(predicted[0], time, event)
        reconstruction_loss = torch.nn.MSELoss()(predicted[1], predicted[2])
        return alpha * cox_loss + reconstruction_loss
