import scipy
import torch
from skorch.callbacks import EarlyStopping
from survboard.python.model.fusion import (
    EarlyFusion,
    IntermediateFusionAttention,
    IntermediateFusionConcat,
    IntermediateFusionMax,
    IntermediateFusionMean,
    LateFusionMean,
)
from survboard.python.model.skorch_infra import (
    CoxPHNeuralNet,
    CoxPHNeuralSGLNet,
    DiscreteTimeNeuralNet,
    DiscreteTimeNeuraSGLNet,
    EHNeuralNet,
    EHNeuralSGLNet,
)
from survboard.python.utils.misc_utils import (
    BreslowLoss,
    BreslowSGLLoss,
    DiscreteTimeLoss,
    DiscreteTimeSGLLoss,
    EHLoss,
    EHSGLLoss,
    StratifiedSkorchSurvivalSplit,
    eh_likelihood_torch_2,
    negative_partial_log_likelihood_loss,
    nll_logistic_hazard_loss,
)

FUSION_FACTORY = {
    "early": EarlyFusion,
    "survival_net": EarlyFusion,
    "late_mean": LateFusionMean,
    "intermediate_mean": IntermediateFusionMean,
    "intermediate_max": IntermediateFusionMax,
    "intermediate_concat": IntermediateFusionConcat,
    "intermediate_attention": IntermediateFusionAttention,
}

CRITERION_FACTORY = {
    "survival_net": BreslowLoss,
    "cox": BreslowLoss,
    "eh": EHLoss,
    "discrete_time": DiscreteTimeLoss,
    "cox_sgl": BreslowSGLLoss,
    "eh_sgl": EHSGLLoss,
    "discrete_time_sgl": DiscreteTimeSGLLoss,
}

HYPERPARAM_FACTORY = {
    "common_fixed": {
        "optimizer": torch.optim.AdamW,
        "max_epochs": 100,
        "train_split": StratifiedSkorchSurvivalSplit(5, stratified=True),
        "verbose": 0,
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
    "early_tuned": {
        "module__lamb": [0, 0.0005, 0.005, 0.05, 0.1],
        "module__alpha": [0.0],
    },
    "late_mean_tuned": {},
    "intermediate_mean_tuned": {},
    "intermediate_max_tuned": {},
    "intermediate_concat_tuned": {},
    "intermediate_attention_tuned": {},
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
    "intermediate_attention_fixed": {
        "module__log_hazard_hidden_layer_size": 64,
        "module__log_hazard_hidden_layers": 1,
        "module__modality_dimension": 128,
    },
    "survival_net_tuned": {
        "optimizer": torch.optim.Adam,
        "optimizer__weight_decay": [0.0],
        "batch_size": [1024],
        "module__p_dropout": scipy.stats.uniform(loc=0.0, scale=0.9),
        "module__log_hazard_hidden_layers": [1, 2, 3, 4, 5],
        "module__log_hazard_hidden_layer_size": [i for i in range(10, 1000)],
    },
}

SKORCH_NET_FACTORY = {
    "survival_net": CoxPHNeuralNet,
    "cox": CoxPHNeuralNet,
    "eh": EHNeuralNet,
    "discrete_time": DiscreteTimeNeuralNet,
    "cox_sgl": CoxPHNeuralSGLNet,
    "eh_sgl": EHNeuralSGLNet,
    "discrete_time_sgl": DiscreteTimeNeuraSGLNet,
}


LOSS_FACTORY = {
    "survival_net": negative_partial_log_likelihood_loss,
    "cox": negative_partial_log_likelihood_loss,
    "eh": eh_likelihood_torch_2,
    "discrete_time": nll_logistic_hazard_loss,
}
