from operator import iconcat
import os
import random
import csv
from unicodedata import category
import warnings
from xml.etree.ElementInclude import include
from numpy import dtype
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

from typing import List, Tuple

# TODO: transform categorical clinical varibales to numeric so can convert to tensor
# TODO: OR write a custom collate function for batching
#


class MultimodalDataset(Dataset):
    """Dataset class for MultiSurv; Returns a dictionary where each key is a modality
    and the corresponding value is the tensor
    """

    def __init__(
        self,
        data_path: str,
        label_path: str = None,
        modalities: List[str] = ["clinical", "gex", "mirna", "cnv", "meth", "mut"],
        dropout: int = 0,
        device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    ) -> None:
        super().__init__()

        self.data = pd.read_csv(data_path, index_col=0)
        if label_path:
            self.labels = pd.read_csv(label_path)
        else:
            try:
                self.labels = self.data[["OS_days", "OS"]]
            except KeyError:
                print("Survival event and time not available in data. Please provide a path to label file instead.")

        try:
            self.patient_ids = self.data["patient_id"]
        except KeyError:
            print("patient_id not found in data, using index")
            self.patient_ids = self.data.index

        self.available_modalities = [m for m in modalities if any(self.data.columns.str.contains(m))]

        assert 0 <= dropout <= 1, '"dropout" must be in [0, 1].'
        self.dropout = dropout

        assert all(
            any(self.data.columns.str.contains(m)) for m in self.available_modalities
        ), "One or more modalities not present in the data"
        # assert all(any(self.data.columns.str.contains(m)) for m in self.available_modalities), "One or more modalities not present in the data"

    def _get_modality(self, modality, patient_id):
        columns_to_subset = self.data.columns[self.data.columns.str.contains(modality)]
        subset = self.data.loc[patient_id, columns_to_subset]

        if modality == "clinical":
            # return torch.zeros(1)
            # TODO: add a transformation here for clinical -> tensor
            return subset.to_numpy()
        elif all(subset.isna()):
            return self._set_missing_modality(subset)
        else:
            return torch.from_numpy(np.array(subset, dtype=np.float32))

    def _clinical_to_tuple(self, clinical):
        categorical = clinical.select_dtypes(include=[object])
        continuous = clinical.select_dtypes(include=[int, float])

        return categorical, continuous

    def _set_missing_modality(self, data, value: float = 0.0):

        return torch.from_numpy(np.array(data.fillna(value), dtype=np.float32))

    def _drop_data(self, data):

        # for clinical, multisurv only uses continous features for drop out

        # Drop data modality
        n_mod = len(self.available_modalities)
        modalities_to_drop = self.available_modalities
        modalities_to_drop.remove("clinical")
        if n_mod > 1:
            if random.random() < self.dropout:
                drop_modality = random.choice(modalities_to_drop)

                data[drop_modality] = torch.zeros_like(data[drop_modality])

        return data

    def get_patient_dict(self, patient_id):
        time, event = self.labels.loc[patient_id]
        data = {}

        # Load selected patient's data
        for modality in self.available_modalities:
            data[modality] = self._get_modality(modality, patient_id)

        # Data dropout
        if self.dropout > 0:
            n_modalities = len([k for k in data])
            if n_modalities > 1:
                data = self._drop_data(data)

        return data, time, event

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        data, time, event = self.get_patient_dict(patient_id)
        # target = np.array([f"{int(i[0])}|{i[1]}" for i in target])
        return data, (time, event)
