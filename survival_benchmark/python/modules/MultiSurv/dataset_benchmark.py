import os
import random
import csv
import warnings
import pandas as pd 

import torch
from torch.utils.data import Dataset

from typing import List, Tuple

class MultimodalDataset(Dataset):
    """Dataset class for MultiSurv; Returns a dictionary where each key is a modality
    and the corresponding value is the tensor 
    """

    def __init__(self, data_path:str,label_path:str, modalities:List[str], dropout:int=0, device:torch.device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')) -> None:
        super().__init__()

        self.data = pd.read_csv(data_path,sep='\t')
        self.labels = pd.read_csv(label_path)

        self.modalities = modalities

        assert all(any(self.data.columns.str.contains(m)) for m in modalities), "One or more modalities not present in the data"

    
    def _get_modality(self, modality):
        columns_to_subset = self.data.columns[self.data.columns.str.contains(modality)]

        return self.data[columns_to_subset]
    
    def _get_patient_ids(self):
        return self.data['patient_id']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pass 