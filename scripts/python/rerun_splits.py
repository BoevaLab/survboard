import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold


def main() -> int:
    config_path = "./config/"
    data_dir = "./"
    with open(f"{config_path}/config.json") as f:
        config = json.load(f)
    np.random.seed(config["random_state"])
    for project in ["TCGA", "ICGC", "TARGET", "METABRIC"]:
        Path(f"./data_reproduced/splits/{project}/").mkdir(parents=True, exist_ok=True)
        for cancer in config[f"{project.lower()}_cancers"]:
            print(cancer)
            data_path = f"./data_reproduced/{project}/{cancer}_data_complete_modalities_preprocessed.csv"
            data = pd.read_csv(
                os.path.join(data_dir, data_path),
                low_memory=False,
            )

            # Exact column choice doesn't matter
            # as this is only to create the splits anyway.
            X = data[[i for i in data.columns if i not in ["OS_days", "OS"]]]
            cv = RepeatedStratifiedKFold(
                n_repeats=config["outer_repetitions"],
                n_splits=config["outer_splits"],
                random_state=config["random_state"],
            )
            splits = [i for i in cv.split(X, data["OS"])]
            pd.DataFrame([i[0] for i in splits]).to_csv(
                f"./data_reproduced/splits/{project}/{cancer}_train_splits.csv",
                index=False,
            )
            pd.DataFrame([i[1] for i in splits]).to_csv(
                f"./data_reproduced/splits/{project}/{cancer}_test_splits.csv",
                index=False,
            )
    return 0


if __name__ == "__main__":
    main()
