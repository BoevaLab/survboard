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
    transform_discrete_time,
)

parser = argparse.ArgumentParser()

parser.add_argument(
    "--fusion_choice",
    type=str,
)

parser.add_argument(
    "--model_type_choice",
    type=str,
)


def main(fusion_choice: str, model_type_choice: str):
    with open(os.path.join("./config/", "config.json"), "r") as f:
        config = json.load(f)

    g = np.random.default_rng(config.get("random_state"))
    seed_torch(config.get("random_state"))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    for fusion in [fusion_choice]:
        for model_type in [model_type_choice]:
            for project in ["validation"]:
                for cancer in ["PAAD", "LIHC"]:
                    if cancer == "PAAD":
                        data_path_input = (
                            f"./data_reproduced/{project}/{cancer}/icgc_paca_ca.csv"
                        )
                        data_path_transfer_to = (
                            f"./data_reproduced/{project}/{cancer}/tcga_paad.csv"
                        )
                        split_project = "ICGC"
                        split_cancer = "PACA-CA"
                    elif cancer == "LIHC":
                        data_path_input = (
                            f"./data_reproduced/{project}/{cancer}/tcga_lihc.csv"
                        )
                        data_path_transfer_to = (
                            f"./data_reproduced/{project}/{cancer}/icgc_liri_jp.csv"
                        )
                        split_project = "TCGA"
                        split_cancer = "LIHC"
                    input_data = pd.read_csv(
                        os.path.join(data_path_input),
                        low_memory=False,
                    )
                    test_data = pd.read_csv(
                        os.path.join(data_path_transfer_to),
                        low_memory=False,
                    )
                    data_helper = input_data.copy(deep=True).drop(
                        columns=["OS_days", "OS"]
                    )
                    train_splits = pd.read_csv(
                        os.path.join(
                            f"./data_reproduced/splits/{split_project}/{split_cancer}_train_splits.csv"
                        )
                    )
                    for outer_split in range(
                        config["outer_repetitions"] * config["outer_splits"]
                    ):
                        train_ix = (
                            train_splits.iloc[outer_split, :]
                            .dropna()
                            .values.astype(int)
                        )
                        X_test = test_data.reset_index(drop=True)
                        X_train = (
                            input_data.iloc[train_ix, :]
                            .sort_values(by="OS_days", ascending=True)
                            .reset_index(drop=True)
                        )
                        time_train = X_train["OS_days"].values
                        event_train = X_train["OS"].values

                        X_train = X_train.drop(columns=["OS", "OS_days"])
                        X_test = X_test.drop(columns=["OS", "OS_days"])
                        time_discretized, cuts = transform_discrete_time(
                            time=time_train,
                            event=event_train,
                            n_durations=config["n_discrete_durations"],
                        )
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
                        if model_type == "discrete_time":
                            time_train = time_discretized[0]
                            event_train = time_discretized[1]
                        y_train = transform(time_train, event_train)
                        X_train = ct.fit_transform(X_train)

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
                        X_test = pd.DataFrame(
                            ct.transform(X_test), columns=X_train.columns
                        )
                        if fusion == "early":
                            factory_model_type = f"{model_type}_sgl"
                        else:
                            factory_model_type = model_type
                        net = SKORCH_NET_FACTORY[factory_model_type](
                            module=(
                                SKORCH_MODULE_FACTORY[model_type]
                                if fusion != "early"
                                else SKORCH_MODULE_FACTORY[f"{model_type}_sgl"]
                            ),
                            criterion=(
                                CRITERION_FACTORY[model_type]
                                if fusion != "early"
                                else CRITERION_FACTORY[f"{model_type}_sgl"]
                            ),
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
                        hyperparams = HYPERPARAM_FACTORY["common_tuned"].copy()
                        if fusion == "early":
                            hyperparams.update(HYPERPARAM_FACTORY["early_tuned"])
                        grid = RandomizedSearchCV(
                            net,
                            hyperparams,
                            n_jobs=10,
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
                            pre_dispatch=10,
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
                        elif model_type == "eh" and success:
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
                        elif model_type == "discrete_time" and success:
                            try:
                                sf_df = grid.best_estimator_.predict_survival_function(
                                    X_test.to_numpy().astype(np.float32), cuts
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

                        pathlib.Path(
                            f"./results_reproduced/survival_functions/transfer/{project}/{cancer}/{model_type}_{fusion}/"
                        ).mkdir(parents=True, exist_ok=True)

                        sf_df.to_csv(
                            f"./results_reproduced/survival_functions/transfer/{project}/{cancer}/{model_type}_{fusion}/split_{outer_split}.csv",
                            index=False,
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
    main(args.fusion_choice, args.model_type_choice)
