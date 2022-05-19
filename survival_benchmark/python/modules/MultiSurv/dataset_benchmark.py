import os
import random
from numpy import dtype
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
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
        modalities: List[str] = ["clinical", "gex", "mirna", "cnv", "meth", "mut", "rppa"],
        dropout: int = 0,
        device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    ) -> None:
        """Constructor.

        Args:
            data_path (str): Path to the multimodal merged data in csv format.
            label_path (str, optional): Path to the file containing event and event time in csv format.
            Defaults to None.
            modalities (List[str], optional): List of modalities considered.
            Must be in the same form as used in naming columns in the merged file.
            Defaults to ["clinical", "gex", "mirna", "cnv", "meth", "mut"].
            dropout (int, optional): Modality drop out probability. Defaults to 0.
            device (_type_, optional): Device on which experiments are run. Defaults to torch.device("cuda:0" if torch.cuda.is_available() else "cpu").
        """
        super().__init__()

        self.data = pd.read_csv(data_path, index_col=0)
        self.input_size = {}
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

    def get_input_size(self, modality, size):
        self.input_size.update({modality: size})

    def _get_modality(self, modality: str, patient_id: str) -> torch.Tensor:
        """Retrieve modality specific features from the merged data file.

        Args:
            modality (str): Modality to retrieve.
            patient_id (str): Patient ID for which modality information should be retrieved.

        Returns:
            torch.Tensor: Tensor of the modality for a given patient. If NA, then returns
                a tensor of zeroes (default value, can be changed).
        """
        columns_to_subset = self.data.columns[self.data.columns.str.contains(modality)]
        self.get_input_size(modality, len(columns_to_subset))
        subset = self.data.loc[patient_id, columns_to_subset]

        if modality == "clinical":
            categorical, continuous = self._clinical_to_tuple(subset)
            return subset.to_numpy()
        elif all(subset.isna()):
            return self._set_missing_modality(subset)
        else:
            return torch.from_numpy(np.array(subset, dtype=np.float32))

    def _clinical_to_tuple(self, clinical: pd.DataFrame) -> Tuple:
        """Breaks down clinical data into continous and categorical variables.

        Args:
            clinical (pd.DataFrame): Dataframe of clinical data.

        Returns:
            Tuple: Tuple of dataframes separating categorical from continuous.
        """
        cat_encoder = OrdinalEncoder()
        categorical = clinical.select_dtypes(include=[object])
        continuous = clinical.select_dtypes(include=[int, float])

        categorical = torch.tensor(cat_encoder.fit_transform(categorical), dtype=torch.int)
        continuous = torch.from_numpy(continuous.to_numpy())

        return categorical, continuous

    def _set_missing_modality(self, data: pd.DataFrame, value: float = 0.0) -> torch.Tensor:
        """Sets values of missing modalities.

        Args:
            data (pd.DataFrame): Dataframe of patient omics data.
            value (float, optional): Value to replace NA. Defaults to 0.0.

        Returns:
            torch.Tensor: Tensor of missing modality features filled with a specified value.
        """

        return torch.from_numpy(np.array(data.fillna(value), dtype=np.float32))

    def _drop_data(self, data: dict) -> dict:
        """Randomly drop a modality based on a dropout probability.

        Args:
            data (dict): Dictionary wherein keys are modalities and their corresponding
                values are tensors (for multi-omics) and np.ndarray (for clinical) of that
                modality for a given patient.

        Returns:
            dict: Dictionary wherein the chosen modality to be dropped is set to zeroes.
        """
        available_modalities = []
        # Check available modalities in current mini-batch
        for modality, values in data.items():
            if isinstance(values, (list, tuple)):  # Clinical data
                values = values[1]  # Use continuous features
            if len(torch.nonzero(values)) > 0:  # Keep if data is available
                available_modalities.append(modality)

        # Drop data modality
        n_mod = len(available_modalities)

        if n_mod > 1:
            if random.random() < self.dropout:
                drop_modality = random.choice(self.available_modalities)
                if isinstance(data[drop_modality], (list, tuple)):
                    # Clinical data
                    data[drop_modality] = tuple(torch.zeros_like(x) for x in data[drop_modality])
                else:
                    data[drop_modality] = torch.zeros_like(data[drop_modality])

        return data

    def get_patient_dict(self, patient_id: str) -> Tuple:
        """Create dictionary for each patient with modality information.

        Args:
            patient_id (str): ID of the patient for whom a dictionary is created.

        Returns:
            Tuple: Tuple of the patient dictionary, time to event, and event observed.
        """
        time, event = self.labels.loc[patient_id]
        data = {}

        # Load selected patient's data
        for modality in self.available_modalities:
            data[modality] = self._get_modality(modality, patient_id)
            if isinstance(data[modality], (list, tuple)):  # Clinical data
                data[modality] = tuple(x.float() for x in data[modality])

        # Data dropout
        if self.dropout > 0:
            n_modalities = len([k for k in data])
            if n_modalities > 1:
                data = self._drop_data(data)

        return data, time, event

    def __len__(self) -> int:
        """Get length of the dataset.

        Returns:
            int: length of the merged dataset, i.e, total number of patients.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple:
        """Samples a patient from the merged dataset.

        Args:
            idx (int): Index of the patient sample.

        Returns:
            Tuple: Tuple containing patient dictionary and a second tuple of the event and time.
        """
        patient_id = self.patient_ids[idx]
        data, time, event = self.get_patient_dict(patient_id)
        # target = np.array([f"{int(event)}|{time}"])
        return data, (time, event)
