import torch
import torch.nn as nn
from survival_benchmark.python.utils.hyperparameters import ACTIVATION_FN_FACTORY


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

        """
        super(FCBlock, self).__init__()

        self.input_size = params.get("input_size", 256)
        self.latent_dim = params.get("latent_dim", 64)

        self.hidden_size = params.get("fc_units", [128, 64])
        self.layers = params.get("fc_layers", 2)
        self.scaling_factor = eval(params.get("scaling_factor", 0.5))

        self.activation = params.get("fc_activation", ["relu", "None"])
        self.dropout = params.get("fc_dropout", 0.5)
        self.batchnorm = eval(params.get("fc_batchnorm", "False"))

        if len(self.hidden_size) != self.layers and self.reduction_factor is not None:
            hidden_size_generated = []
            # factor = (self.reduction_factor - 1) / self.reduction_factor
            for layer in range(self.layers):
                try:
                    hidden_size_generated.append(self.hidden_size[layer])
                except IndexError:
                    if layer == self.layers - 1:
                        hidden_size_generated.append(self.latent_dim)
                    else:
                        hidden_size_generated.append(hidden_size_generated[-1] * self.scaling_factor)

            self.hidden_size = hidden_size_generated

        if len(self.activation) != self.layers:
            if len(self.activation) == 2:
                first, last = self.activation
                self.activation = [first] * len(self.layers - 1) + [last]

            elif len(self.activation) == 1:
                self.activation = self.activation * len(self.layers)

            else:
                raise ValueError

        modules = []
        self.hidden_units = [self.input_size] + self.hidden_size
        for layer in range(self.layers):
            modules.append(nn.Linear(self.hidden_units[layer], self.hidden_units[layer + 1]))
            if self.activation[layer] != "None":
                modules.append(ACTIVATION_FN_FACTORY[self.activation[layer]])
            if self.dropout > 0:
                modules.append(nn.Dropout(self.dropout))
            if self.batchnorm:
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

        # update based on enc params in AE class
        # self.dec_params.update(
        #     {
        #         "input_size": self.latent_dim,
        #         "latent_dim": self.input_size,
        #         "fc_units": self.hidden_units[-2::-1],
        #     }
        # )
        self.decoder = FCBlock(self.dec_params)

    def forward(self, x):
        return self.decoder(x)
