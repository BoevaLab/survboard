import torch
import torch.nn as nn
from survival_benchmark.python.utils.hyperparameters import ACTIVATION_FN_FACTORY

from survival_benchmark.python.modules.modules import HazardRegression
from survival_benchmark.python.utils.utils import cox_criterion


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
        self.scaling_factor = eval(params.get("scaling_factor", 0.5))

        self.activation = params.get("fc_activation", ["relu", "None"])
        self.dropout = params.get("fc_dropout", 0.5)
        self.batchnorm = eval(params.get("fc_batchnorm", "False"))
        self.bias_last = eval(params.get("last_layer_bias","True"))
        bias = [True]*(self.layers-1) + [self.bias_last]

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
            modules.append(nn.Linear(self.hidden_units[layer], self.hidden_units[layer + 1],bias=bias[layer]))
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

class DAE(nn.Module):
    def __init__(self,
        input_dimensionality,
        output_dimensionality,
        noise_factor=0, # TODO: or 0.5??
        activation=nn.PReLU,
        n_hidden_layers=2,
        n_first_hidden_layer=128,
        n_latent_space=64,
        p_dropout=0.5,
        batch_norm=False,
        decrease_factor_per_layer=2,
        alpha=0.1
    ) -> None:
        super().__init__()

        self.alpha = alpha
        self.noise_factor = noise_factor
        
        self.encoder = Encoder(
            n_input=input_dimensionality,
            activation=activation,
            n_hidden_layers=n_hidden_layers,
            n_first_hidden_layer=n_first_hidden_layer,
            n_latent_space=n_latent_space,
            p_dropout=p_dropout,
            batch_norm=batch_norm,
            decrease_factor_per_layer=decrease_factor_per_layer
        )
        
        self.decoder = Decoder(self.encoder)
        
        self.hazard = HazardRegression(
            input_dimension=self.encoder.n_latent_space,
            n_output=output_dimensionality
        )
        
    def forward(self, x):

        x_noisy = x+(self.noise_factor*torch.normal(mean=0.0, std=1, size=x.shape)) 
        encoded = self.encoder(x_noisy)
        decoded = self.decoder(encoded)
        log_hazard = self.hazard(encoded)
        return log_hazard, x, decoded
    
class dae_criterion(nn.Module):
    def forward(self, predicted, target):
        cox_loss = cox_criterion(predicted[0], target)
        reconstruction_loss = torch.nn.MSE(predicted[1], predicted[2])
        return self.alpha * cox_loss + reconstruction_loss 

