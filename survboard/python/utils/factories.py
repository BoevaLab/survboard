import torch
from skorch.callbacks import EarlyStopping

from survboard.python.model.fusion import (
    EarlyFusion,
    IntermediateFusionConcat,
    IntermediateFusionMax,
    IntermediateFusionMean,
    LateFusionMean,
)
from survboard.python.model.skorch_infra import CoxPHNeuralNet, EHNeuralNet
from survboard.python.utils.misc_utils import (
    BreslowLoss,
    EHLoss,
    StratifiedSkorchSurvivalSplit,
    eh_likelihood_torch_2,
    negative_partial_log_likelihood_loss,
)

FUSION_FACTORY = {
    "early": EarlyFusion,
    "late_mean": LateFusionMean,
    "intermediate_mean": IntermediateFusionMean,
    "intermediate_max": IntermediateFusionMax,
    "intermediate_concat": IntermediateFusionConcat,
}

CRITERION_FACTORY = {
    "cox": BreslowLoss,
    "eh": EHLoss,
}

HYPERPARAM_FACTORY = {
    "common_fixed": {
        "optimizer": torch.optim.AdamW,
        "max_epochs": 100,
        "train_split": StratifiedSkorchSurvivalSplit(5, stratified=True),
        "verbose": False,
        "callbacks": [
            (
                "es",
                EarlyStopping(
                    monitor="valid_loss",
                    patience=10,
                    load_best=True,
                ),
            ),
        ],
        "module__activation": torch.nn.ReLU,
    },
    "common_tuned": {
        "lr": [0.0005, 0.0008, 0.001, 0.005],
        "optimizer__weight_decay": [0.0005, 0.005, 0.05, 0.1],
        "module__modality_hidden_layer_size": [32, 64, 128, 256, 512],
        "module__modality_hidden_layers": [1],
        "module__p_dropout": [0, 0.2, 0.4, 0.6],
        "batch_size": [1024],
    },
    "early_tuned": {},
    "late_mean_tuned": {},
    "intermediate_mean_tuned": {},
    "intermediate_max_tuned": {},
    "intermediate_concat_tuned": {},
    "early_fixed": {
        "module__log_hazard_hidden_layer_size": 128,
        "module__log_hazard_hidden_layers": 2,
    },
    "late_mean_fixed": {
        "module__log_hazard_hidden_layer_size": 128,
        "module__log_hazard_hidden_layers": 2,
    },
    "intermediate_concat_fixed": {
        "module__log_hazard_hidden_layer_size": 64,
        "module__log_hazard_hidden_layers": 1,
        "module__modality_dimension": 128,
    },
}

SKORCH_NET_FACTORY = {
    "cox": CoxPHNeuralNet,
    "eh": EHNeuralNet,
}


LOSS_FACTORY = {
    "cox": negative_partial_log_likelihood_loss,
    "eh": eh_likelihood_torch_2,
}
