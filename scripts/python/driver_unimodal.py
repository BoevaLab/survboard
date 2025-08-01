import argparse
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
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from skorch.callbacks import EarlyStopping
from sksurv.nonparametric import kaplan_meier_estimator

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
    get_blocks,
    get_cumulative_hazard_function_eh,
    seed_torch,
    transform,
)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--project",
    type=str,
)

parser.add_argument(
    "--cancer",
    type=str,
)

parser.add_argument(
    "--split",
    type=int,
)


def main(project: str, cancer: str, split: int):

    with open(os.path.join("./config/", "config.json"), "r") as f:
        config = json.load(f)
    g = np.random.default_rng(config.get("random_state"))
    seed_torch(config.get("random_state"))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    fusion = "early"
    all_sf_dfs_for_task = []
    for model_type in ["eh", "cox"]:
        for modality in ["clinical", "gex", "mirna", "meth", "rppa", "cnv", "mut"]:
            for project in [project]:
                for cancer in [cancer]:
                    if modality not in config[f"{project.lower()}_modalities"][cancer]:
                        continue
                    data_path = f"./data_reproduced/{project}/{cancer}_data_complete_modalities_preprocessed.csv"
                    data = pd.read_csv(
                        os.path.join(data_path),
                        low_memory=False,
                    ).drop(columns=["patient_id"])
                    data = data.iloc[
                        :,
                        [
                            i
                            for i in range(data.shape[1])
                            if data.columns[i].rsplit("_")[0] in [modality, "OS"]
                        ],
                    ]
                    data_helper = data.copy(deep=True).drop(columns=["OS_days", "OS"])
                    constant_columns = data_helper.columns[data_helper.nunique() == 1]

                    data_helper = data_helper.drop(columns=constant_columns)

                    train_splits = pd.read_csv(
                        os.path.join(
                            f"./data_reproduced/splits/{project}/{cancer}_train_splits.csv"
                        )
                    )
                    test_splits = pd.read_csv(
                        os.path.join(
                            f"./data_reproduced/splits/{project}/{cancer}_test_splits.csv"
                        )
                    )
                    data = pd.concat(
                        [
                            data_helper,
                            data.loc[:, (np.isin(data.columns, ["OS", "OS_days"]))],
                        ],
                        axis=1,
                    )

                    for outer_split in [split]:
                        train_ix = (
                            train_splits.iloc[outer_split, :]
                            .dropna()
                            .values.astype(int)
                        )
                        test_ix = (
                            test_splits.iloc[outer_split, :].dropna().values.astype(int)
                        )
                        X_test = data.iloc[test_ix, :].reset_index(drop=True)
                        X_train = (
                            data.iloc[train_ix, :]
                            .sort_values(by="OS_days", ascending=True)
                            .reset_index(drop=True)
                        )
                        time_train = X_train["OS_days"].values
                        time_test = X_test["OS_days"].values
                        event_train = X_train["OS"].values
                        event_test = X_test["OS"].values

                        X_train = X_train.drop(columns=["OS", "OS_days"])
                        X_test = X_test.drop(columns=["OS", "OS_days"])

                        if modality == "clinical":
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
                                            StandardScaler(),
                                        ),
                                        np.where(X_train.dtypes == "object")[0],
                                    ),
                                ]
                            )
                        else:
                            ct = ColumnTransformer(
                                [
                                    (
                                        "numerical",
                                        make_pipeline(StandardScaler()),
                                        np.where(X_train.dtypes != "object")[0],
                                    )
                                ]
                            )
                        y_train = transform(time_train, event_train)
                        X_train = ct.fit_transform(X_train)
                        if modality == "clinical":
                            X_train = pd.DataFrame(
                                X_train,
                                columns=data_helper.columns[
                                    np.where(data_helper.dtypes != "object")[0]
                                ].tolist()
                                + [
                                    f"clinical_{i}"
                                    for i in ct.transformers_[1][1][0]
                                    .get_feature_names_out()
                                    .tolist()
                                ],
                            )
                        else:
                            X_train = pd.DataFrame(
                                X_train,
                                columns=data_helper.columns[
                                    np.where(data_helper.dtypes != "object")[0]
                                ].tolist(),
                            )
                        X_test = pd.DataFrame(
                            ct.transform(X_test), columns=X_train.columns
                        )

                        net = SKORCH_NET_FACTORY[model_type](
                            module=SKORCH_MODULE_FACTORY[model_type],
                            criterion=CRITERION_FACTORY[model_type],
                            module__fusion_method=fusion,
                            module__blocks=get_blocks(X_train.columns),
                            iterator_train__shuffle=True,
                        )
                        net.set_params(
                            **HYPERPARAM_FACTORY["common_fixed"],
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
                        grid = RandomizedSearchCV(
                            net,
                            HYPERPARAM_FACTORY["common_tuned"],
                            n_jobs=1,
                            cv=StratifiedSurvivalKFold(
                                n_splits=5,
                                shuffle=True,
                                random_state=config.get("random_state"),
                            ),
                            scoring=make_scorer(
                                LOSS_FACTORY[model_type],
                                greater_is_better=False,
                            ),
                            error_score=100,
                            verbose=0,
                            n_iter=50,
                            random_state=42,
                        )

                        try:
                            grid.fit(X_train.to_numpy().astype(np.float32), y_train)
                            success = True
                        except ValueError:
                            success = False
                        if model_type == "cox" and success:
                            survival_functions = (
                                grid.best_estimator_.predict_survival_function(
                                    X_test.to_numpy().astype(np.float32)
                                )
                            )
                            survival_probabilities = np.stack(
                                [
                                    i(np.unique(np.abs(y_train)))
                                    for i in survival_functions
                                ]
                            )

                            sf_df = pd.DataFrame(
                                survival_probabilities,
                                columns=np.unique(np.abs(y_train)),
                            )
                        elif success:
                            try:
                                sf_df = np.exp(
                                    np.negative(
                                        get_cumulative_hazard_function_eh(
                                            None,
                                            None,
                                            y_train,
                                            None,
                                            grid.best_estimator_.predict(
                                                X_train.to_numpy().astype(np.float32)
                                            )
                                            .detach()
                                            .numpy(),
                                            grid.best_estimator_.predict(
                                                X_test.to_numpy().astype(np.float32)
                                            )
                                            .detach()
                                            .numpy(),
                                        )
                                    )
                                )
                            except ValueError:
                                sf_df = np.nan
                        else:
                            sf_df = np.nan
                        if np.any(np.isnan(sf_df)):
                            time_km, survival_km = kaplan_meier_estimator(
                                event_train.astype(bool),
                                time_train,
                            )
                            sf_df = pd.DataFrame(
                                [survival_km for i in range(X_test.shape[0])]
                            )
                            sf_df.columns = time_km
                        sf_df["model_type"] = model_type
                        sf_df["modality"] = modality
                        sf_df["project"] = project
                        sf_df["cancer"] = cancer
                        sf_df["split"] = outer_split

                        all_sf_dfs_for_task.append(sf_df)

    if all_sf_dfs_for_task:
        consolidated_sf_df = pd.concat(all_sf_dfs_for_task, ignore_index=True)

        output_dir = pathlib.Path(
            f"./results_reproduced/survival_functions_consolidated_csv/{project}/{cancer}/unimodal"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file_path = output_dir / f"split_{split}.csv"
        consolidated_sf_df.to_csv(output_file_path, index=False)
        print(
            f"Saved consolidated results for {project}/{cancer}/split_{split} to {output_file_path}"
        )
    else:
        print(
            f"No data generated for {project}/{cancer}/split_{split}. No CSV file saved."
        )
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


if __name__ == "__main__":
    args = parser.parse_args()
    main(args.project, args.cancer, args.split)
