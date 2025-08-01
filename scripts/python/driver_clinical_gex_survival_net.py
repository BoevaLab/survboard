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
from sklearn.feature_selection import VarianceThreshold
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


def filter_modality(data, quantile):
    data_var = data.apply(np.var, axis=0)
    data_var_quantile = np.quantile(data_var, quantile)
    chosen_cols = data.iloc[:, np.where(data_var > data_var_quantile)[0]].columns
    return chosen_cols


def main(project: str, cancer: str, split: int):
    with open(os.path.join("./config/", "config.json"), "r") as f:
        config = json.load(f)

    g = np.random.default_rng(config.get("random_state"))
    seed_torch(config.get("random_state"))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    for fusion in ["survival_net"]:
        for model_type in ["survival_net"]:
            for project in [project]:
                for cancer in [cancer]:
                    data_path = f"./data_reproduced/{project}/{cancer}_data_complete_modalities_preprocessed.csv"
                    data = pd.read_csv(
                        os.path.join(data_path),
                        low_memory=False,
                    ).drop(columns=["patient_id"])

                    feature_names = data.columns
                    column_types = (
                        pd.Series(feature_names)
                        .str.rsplit("_")
                        .apply(lambda x: x[0])
                        .values
                    )
                    mask = np.isin(column_types, ["clinical", "gex", "OS"])
                    data = data.loc[:, mask]
                    feature_names = data.columns
                    column_types = (
                        pd.Series(feature_names)
                        .str.rsplit("_")
                        .apply(lambda x: x[0])
                        .values
                    )
                    survival_data = data.iloc[:, np.where(column_types == "OS")[0]]
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

                    available_modalities = [
                        i for i in np.unique(column_types) if i != "OS"
                    ]
                    clinical_cols = data.iloc[
                        :, np.where(np.isin(column_types, ["OS", "clinical"]))[0]
                    ].columns.tolist()
                    chosen_cols = []
                    for modality in [
                        i for i in available_modalities if i != "clinical"
                    ]:
                        if modality in ["gex", "meth", "mut", "cnv"]:
                            chosen_cols += filter_modality(
                                data.iloc[
                                    :, np.where(np.isin(column_types, [modality]))[0]
                                ],
                                0.9,
                            ).tolist()
                        else:
                            chosen_cols += filter_modality(
                                data.iloc[
                                    :, np.where(np.isin(column_types, [modality]))[0]
                                ],
                                0.0,
                            ).tolist()

                    chosen_cols = clinical_cols + chosen_cols

                    data_overall = data[chosen_cols]
                    data_overall_vars = data_overall.loc[
                        :,
                        np.logical_not(
                            np.isin(data_overall.columns, ["OS", "OS_days"])
                        ),
                    ]

                    ct = ColumnTransformer(
                        [
                            (
                                "numerical",
                                make_pipeline(StandardScaler()),
                                np.where(data_overall_vars.dtypes != "object")[0],
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
                                np.where(data_overall_vars.dtypes == "object")[0],
                            ),
                        ]
                    )
                    data_vars_overall_transformed = pd.DataFrame(
                        ct.fit_transform(data_overall_vars),
                        columns=(
                            ct.transformers_[0][1][0].get_feature_names_out().tolist()
                            + [
                                f"clinical_{i}"
                                for i in ct.transformers_[1][1][1]
                                .get_feature_names_out()
                                .tolist()
                            ]
                        ),
                    )
                    data_finalized = pd.concat(
                        [
                            data_vars_overall_transformed,
                            data_overall.loc[
                                :, (np.isin(data_overall.columns, ["OS", "OS_days"]))
                            ],
                        ],
                        axis=1,
                    )

                    data_helper = data.copy(deep=True).drop(columns=["OS_days", "OS"])
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
                    for outer_split in [split]:
                        train_ix = (
                            train_splits.iloc[outer_split, :]
                            .dropna()
                            .values.astype(int)
                        )
                        test_ix = (
                            test_splits.iloc[outer_split, :].dropna().values.astype(int)
                        )
                        X_test = data_finalized.iloc[test_ix, :].reset_index(drop=True)
                        X_train = (
                            data_finalized.iloc[train_ix, :]
                            .sort_values(by="OS_days", ascending=True)
                            .reset_index(drop=True)
                        )
                        time_train = X_train["OS_days"].values
                        time_test = X_test["OS_days"].values
                        event_train = X_train["OS"].values
                        event_test = X_test["OS"].values

                        X_train = X_train.drop(columns=["OS", "OS_days"])
                        X_test = X_test.drop(columns=["OS", "OS_days"])

                        y_train = transform(time_train, event_train)

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
                            **HYPERPARAM_FACTORY["common_fixed_survival_net"],
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
                        hyperparams = HYPERPARAM_FACTORY["survival_net_tuned"].copy()
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
                        except ValueError as e:
                            raise e
                            success = False
                        if model_type == "survival_net" and success:
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
                            f"./results_reproduced/survival_functions/clinical_gex/{project}/{cancer}/{model_type}_{fusion}/"
                        ).mkdir(parents=True, exist_ok=True)

                        sf_df.to_csv(
                            f"./results_reproduced/survival_functions/clinical_gex/{project}/{cancer}/{model_type}_{fusion}/split_{outer_split}.csv",
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
    main(args.project, args.cancer, args.split)
