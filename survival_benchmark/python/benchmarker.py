import os
import sys
import numpy as np 
import pandas as pd
import skorch
import torch
import torch.nn as nn 

class SurvivalBenchmark(nn.Module):
    def __init__(self, model_params) -> None:
        super().__init__()

        pass

    def forward(self,dataset, model):
        pass
