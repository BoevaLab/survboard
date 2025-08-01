import json
import os
import pathlib
import platform
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from skorch.callbacks import EarlyStopping

from survboard.python.model.model import SKORCH_MODULE_FACTORY
from survboard.python.model.skorch_infra import FixSeed
from survboard.python.utils.factories import (
    CRITERION_FACTORY,
    HYPERPARAM_FACTORY,
    LOSS_FACTORY,
    SKORCH_NET_FACTORY,
)
from survboard.python.utils.misc_utils import (
    StratifiedSkorchSurvivalSplit,
    StratifiedSurvivalKFold,
    get_blocks_salmon,
    seed_torch,
    transform,
)

with open(snakemake.log[0], "w") as f:
    sys.stderr = f
    sys.stdout = f
    with open(os.path.join("./config/", "config.json"), "r") as f:
        config = json.load(f)

    g = np.random.default_rng(config.get("random_state"))
    seed_torch(config.get("random_state"))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    for fusion in ["salmon"]:
        for model_type in ["salmon"]:
            for project in ["TCGA"]:
                for cancer in [snakemake.wildcards["cancer"]]:
                    data = pd.read_csv(
                        f"./data_reproduced/{project}/{cancer}_salmon_preprocessed.csv",
                        low_memory=False,
                        sep="\t",
                    )
                    feature_names = data.columns
                    column_types = (
                        pd.Series(feature_names)
                        .str.rsplit("_")
                        .apply(lambda x: x[0])
                        .values
                    )
                    mask = np.isin(column_types, ["clinical", "gex", "OS"])
                    data = data.loc[:, mask]

                    train_ix = np.array([i for i in range(data.shape[0])])
                    X_train = (
                        data.iloc[train_ix, :]
                        .sort_values(by="OS_days", ascending=True)
                        .reset_index(drop=True)
                    )

                    time_train = X_train["OS_days"].values
                    event_train = X_train["OS"].values

                    X_train = X_train.drop(columns=["OS", "OS_days"])
                    ct = ColumnTransformer(
                        [
                            (
                                "numerical",
                                make_pipeline(StandardScaler()),
                                np.where(X_train.dtypes != "object")[0],
                            ),
                            (
                                "categorical",
                                make_pipeline(
                                    OneHotEncoder(
                                        sparse=False, handle_unknown="ignore"
                                    ),
                                    VarianceThreshold(threshold=0.01),
                                    StandardScaler(),
                                ),
                                np.where(X_train.dtypes == "object")[0],
                            ),
                        ]
                    )
                    y_train = transform(time_train, event_train)
                    X_train = ct.fit_transform(X_train)

                    X_train = pd.DataFrame(
                        X_train,
                        columns=(
                            ct.transformers_[0][1][0].get_feature_names_out().tolist()
                        )
                        + [
                            f"clinical_{i}"
                            for i in ct.transformers_[1][1][1]
                            .get_feature_names_out()
                            .tolist()
                        ],
                    )
                    available_modalities = np.unique(
                        pd.Series(X_train.columns)
                        .str.rsplit("_")
                        .apply(lambda x: x[0])
                        .values
                    )

                    salmon_blocks = get_blocks_salmon(X_train.columns)
                    if (len(salmon_blocks) - 1) == 4:
                        modality_hidden_layer_sizes = [8, 4, 4, 8]
                    elif (len(salmon_blocks) - 1) == 3 and (
                        "mirna" not in available_modalities
                        or "rppa" not in available_modalities
                    ):
                        modality_hidden_layer_sizes = [8, 4, 8]
                    elif (
                        len(salmon_blocks) - 1
                    ) == 3 and "meth" not in available_modalities:
                        modality_hidden_layer_sizes = [8, 4, 4]
                    elif (len(salmon_blocks) - 1) == 2 and (
                        "mirna" in available_modalities
                        or "rppa" in available_modalities
                    ):
                        modality_hidden_layer_sizes = [8, 4]
                    elif (len(salmon_blocks) - 1) == 2 and (
                        "meth" in available_modalities
                    ):
                        modality_hidden_layer_sizes = [8, 8]
                    elif len(salmon_blocks) - 1 == 1:
                        modality_hidden_layer_sizes = [8]

                    net = SKORCH_NET_FACTORY["salmon"](
                        module=(SKORCH_MODULE_FACTORY[model_type]),
                        criterion=(CRITERION_FACTORY["salmon"]),
                        module__blocks=salmon_blocks,
                        module__modality_hidden_layer_sizes=modality_hidden_layer_sizes,
                        iterator_train__shuffle=True,
                    )
                    net.set_params(
                        **HYPERPARAM_FACTORY["common_fixed_salmon"],
                    )

                    net.set_params(
                        **{
                            "train_split": StratifiedSkorchSurvivalSplit(
                                0.2,
                                stratified=True,
                                random_state=config.get("random_state"),
                            ),
                            "callbacks": [
                                (
                                    "es",
                                    EarlyStopping(
                                        monitor="valid_loss",
                                        patience=10,
                                        load_best=True,
                                    ),
                                ),
                                ("seed", FixSeed(generator=g)),
                            ],
                        }
                    )
                    hyperparams = HYPERPARAM_FACTORY["salmon_tuned"].copy()
                    grid = RandomizedSearchCV(
                        net,
                        hyperparams,
                        n_jobs=1,
                        cv=StratifiedSurvivalKFold(
                            n_splits=5,
                            shuffle=True,
                            random_state=config.get("random_state"),
                        ),
                        scoring=make_scorer(
                            LOSS_FACTORY["cox"],
                            greater_is_better=False,
                        ),
                        error_score=100,
                        verbose=0,
                        n_iter=50,
                        random_state=42,
                    )

                    grid.fit(X_train.to_numpy().astype(np.float32), y_train)
                    pathlib.Path(
                        f"results_reproduced/timings/salmon_{snakemake.wildcards['cancer']}"
                    ).touch()

    print("--- Python Environment ---")
    print(f"Python Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")
    print(
        f"Operating System: {platform.system()} {platform.release()} ({platform.version()})"
    )
    print(f"Architecture: {platform.machine()}")
    print(f"Node Name: {platform.node()}")
    print(f"Current Working Directory: {os.getcwd()}")
    print(f"Platform: {sys.platform}")  # More specific OS identifier
    print(f"Processor: {platform.processor()}")
    print("\n--- Installed Python Packages (pip freeze) ---")
    try:
        result = subprocess.run(
            ["pip", "freeze"], capture_output=True, text=True, check=True
        )
        print(result.stdout)
    except FileNotFoundError:
        print("pip command not found. Make sure pip is installed and in your PATH.")
    except subprocess.CalledProcessError as e:
        print(f"Error running pip freeze: {e}")
