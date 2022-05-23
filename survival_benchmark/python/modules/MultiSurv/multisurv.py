"""Deep Learning-based multimodal data model for survival prediction."""

import warnings
import math
from skorch import NeuralNet
import torch
import numpy as np
import pandas as pd

from survival_benchmark.python.utils.utils import inverse_transform_survival_target
from survival_benchmark.python.modules.modules import BaseSurvivalNeuralNet
from .sub_models import FC, ClinicalNet, CnvNet, Fusion


class MultiSurvModel(NeuralNet):
    def get_loss(self, y_pred, y_true, X=None, training=False):
        modality_features, risk = y_pred
        # time, event = inverse_transform_survival_target(y_true)
        time, event = y_true

        loss = self.criterion_(
            risk=risk,
            times=time,
            events=event,
            breaks=self.module_.output_intervals.double().to(self.device),
            device=self.device,
        )
        return loss

    def _check_dropout(self, dataset):
        dropout = dataset.dropout
        if dropout > 0:
            warnings.warn(f"Data dropout set to {dropout} in input dataset")

    def _convert_to_survival(self, conditional_probabilities):
        return np.cumprod(conditional_probabilities)

    def predict_survival_function(self, data, prediction_year=None, intervals=None):
        """Predict patient survival probability at provided time point.
        intervals - Ex torch.arange(0.5,30,1)
        prediction_year - Ex 0.65 or smth
        if both are None, return surv probability for all intervals
        """
        if prediction_year is not None:
            assert intervals is not None, (
                '"intervals" is required to' + ' compute prediction at a specific "prediction_year".'
            )

        # data = self._clone(patient_data)

        # data = self._data_to_device(data)
        risk = self.predict(data)
        # check this again should be risk i guess
        survival_prob = self._convert_to_survival(risk.cpu())

        if prediction_year is not None:
            survival_prob = np.interp(
                prediction_year * 365,
                intervals,
                # Add probability 1.0 at t0 to match number of intervals
                torch.cat((torch.tensor([1]).float(), survival_prob)),
            )

        return survival_prob

    def predict(self, data):
        # Convert to survival probabilities
        print(data)
        self.module_.eval()

        # with torch.set_grad_enabled(False):
        #     _, risk = self.module_(data)

        return 0


class MultiSurv(torch.nn.Module):
    """Deep Learning model for MULTImodal pan-cancer SURVival prediction."""

    def __init__(self, data_modalities: dict, fusion_method="max", output_intervals=None, device=None):
        super(MultiSurv, self).__init__()
        self.data_modalities = data_modalities.keys()
        self.output_intervals = output_intervals
        n_output_intervals = len(output_intervals) - 1
        self.mfs = 512
        valid_mods = ["clinical", "gex", "mirna", "meth", "cnv", "mut", "rppa"]
        assert all(mod in valid_mods for mod in data_modalities), f"Accepted input data modalitites are: {valid_mods}"

        assert len(data_modalities) > 0, "At least one input must be provided."

        if fusion_method == "cat":
            self.num_features = 0
        else:
            self.num_features = self.mfs

        self.submodels = {}

        # Clinical -----------------------------------------------------------#
        if "clinical" in self.data_modalities:
            embed_dims = list(map(lambda x: (x, int(math.ceil(x / 2))), data_modalities["clinical"]["categorical"]))

            self.clinical_submodel = ClinicalNet(
                output_vector_size=self.mfs,
                embedding_dims=embed_dims,
                n_continuous=data_modalities["clinical"]["continuous"],
            )
            self.submodels["clinical"] = self.clinical_submodel

            if fusion_method == "cat":
                self.num_features += self.mfs

        # RPPA --------------------------------------------------------#
        if "rppa" in self.data_modalities:
            self.rppa_submodel = FC(data_modalities["rppa"], self.mfs, 3)
            self.submodels["rppa"] = self.rppa_submodel

            if fusion_method == "cat":
                self.num_features += self.mfs

        # mRNA ---------------------------------------------------------------#
        if "gex" in self.data_modalities:
            self.gex_submodel = FC(data_modalities["gex"], self.mfs, 3)
            self.submodels["gex"] = self.gex_submodel

            if fusion_method == "cat":
                self.num_features += self.mfs

        # miRNA --------------------------------------------------------------#
        if "mirna" in self.data_modalities:
            self.mirna_submodel = FC(data_modalities["mirna"], self.mfs, 3, scaling_factor=2)
            self.submodels["mirna"] = self.mirna_submodel

            if fusion_method == "cat":
                self.num_features += self.mfs

        # DNAm ---------------------------------------------------------------#
        if "meth" in self.data_modalities:
            self.meth_submodel = FC(data_modalities["meth"], self.mfs, 5, scaling_factor=2)
            self.submodels["meth"] = self.meth_submodel

            if fusion_method == "cat":
                self.num_features += self.mfs

        # CNV ---------------------------------------------------------------#
        if "cnv" in self.data_modalities:
            n_cat = data_modalities["cnv"]["categories"]
            n_embed = int(math.ceil(n_cat / 2)) * data_modalities["cnv"]["length"]
            cnv_embed_dim = [(n_cat, int(math.ceil(n_cat / 2)))] * data_modalities["cnv"]["length"]
            self.cnv_submodel = CnvNet(output_vector_size=self.mfs, embedding_dims=cnv_embed_dim, n_embeddings=n_embed)
            self.submodels["cnv"] = self.cnv_submodel

            if fusion_method == "cat":
                self.num_features += self.mfs

        # Mutation ---------------------------------------------------------------#
        if "mut" in self.data_modalities:
            self.mut_submodel = CnvNet(output_vector_size=self.mfs)
            self.submodels["mut"] = self.mut_submodel

            if fusion_method == "cat":
                self.num_features += self.mfs

        # Instantiate multimodal aggregator ----------------------------------#
        if len(data_modalities) > 1:
            self.aggregator = Fusion(fusion_method, self.mfs, device)
        else:
            if fusion_method is not None:
                warnings.warn("Input data is unimodal: no fusion procedure.")

        # Fully-connected and risk layers ------------------------------------#
        n_fc_layers = 4
        n_neurons = 512

        self.fc_block = FC(in_features=self.num_features, out_features=n_neurons, n_layers=n_fc_layers)

        self.risk_layer = torch.nn.Sequential(
            torch.nn.Linear(in_features=n_neurons, out_features=n_output_intervals), torch.nn.Sigmoid()
        )

    def forward(self, **kwargs):
        # or pass a nested dict (called data) here, and replace **kwargs with 'data'
        multimodal_features = tuple()

        # Run data through modality sub-models (generate feature vectors) ----#
        # for modality in self.data_modalities:
        #     if modality == "clinical":
        #         continue
        #     multimodal_features += (self.submodels[modality](**kwargs[modality]),)

        for modality in self.data_modalities:
            multimodal_features += (self.submodels[modality](kwargs[modality]),)

        # Feature fusion/aggregation -----------------------------------------#
        if len(multimodal_features) > 1:
            x = self.aggregator(torch.stack(multimodal_features))
            feature_repr = {"modalities": multimodal_features, "fused": x}
        else:  # skip if running unimodal data
            x = multimodal_features[0]
            feature_repr = {"modalities": multimodal_features[0]}

        # Outputs ------------------------------------------------------------#
        x = self.fc_block(x)
        risk = self.risk_layer(x)

        # Return non-zero features (not missing input data)
        output_features = tuple()

        for modality in multimodal_features:
            modality_features = torch.stack([batch_element for batch_element in modality if batch_element.sum() != 0])
            output_features += (modality_features,)

        feature_repr["modalities"] = output_features

        return feature_repr, risk
