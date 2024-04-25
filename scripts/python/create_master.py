#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import json
import os

import pandas as pd

with open(os.path.join("./config/", "config.json"), "r") as f:
    config = json.load(f)
config.get("random_state")

for project in ["TCGA", "ICGC", "TARGET", "METABRIC"]:
    for cancer in config[f"{project.lower()}_cancers"]:
        master_file = pd.read_csv(
            f"./data_reproduced/{project}/{cancer}_data_complete_modalities_preprocessed.csv"
        )
        master_file[["OS", "OS_days"]].to_csv(
            f"./data_reproduced/{project}/{cancer}_master.csv", index=False
        )
