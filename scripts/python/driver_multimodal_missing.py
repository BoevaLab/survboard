import argparse
import json
import os
import pathlib

import numpy as np
import pandas as pd
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
    transform,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--fusion",
    type=str,
)


def main(fusion: str):
    with open(os.path.join("./config/", "config.json"), "r") as f:
        config = json.load(f)

    g = np.random.default_rng(config.get("random_state"))

    for fusion in [fusion]:
        for model_type in ["cox", "eh"]:
            for project in ["TCGA"]:
            #for project in ["METABRIC", "TCGA", "ICGC", "TARGET"]:
                #for cancer in config[f"{project.lower()}_cancers"]:
                for cancer in ["KIRP"]:
                    data_path = f"./data_reproduced/{project}/{cancer}_data_complete_modalities_preprocessed.csv"
                    data = pd.read_csv(
                        os.path.join(data_path),
                        low_memory=False,
                    ).drop(columns=["patient_id"])

                    data_path_missing = f"./data_reproduced/{project}/{cancer}_data_incomplete_modalities_preprocessed.csv"
                    data_missing = pd.read_csv(
                        os.path.join(data_path_missing),
                        low_memory=False,
                    ).drop(columns=["patient_id"])
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
                    for outer_split in range(
                        config["outer_repetitions"] * config["outer_splits"]
                    ):
                        train_ix = (
                            train_splits.iloc[outer_split, :]
                            .dropna()
                            .values.astype(int)
                        )
                        test_ix = (
                            test_splits.iloc[outer_split, :].dropna().values.astype(int)
                        )
                        X_test = data.iloc[test_ix, :].reset_index(drop=True)
                        X_train = pd.concat([(
                            data.iloc[train_ix, :]
                            .sort_values(by="OS_days", ascending=True)
                            .reset_index(drop=True)
                        ), data_missing], axis=0)
                        time_train = X_train["OS_days"].values
                        time_test = X_test["OS_days"].values
                        event_train = X_train["OS"].values
                        event_test = X_test["OS"].values

                        X_train = X_train.drop(columns=["OS", "OS_days"])
                        X_test = X_test.drop(columns=["OS", "OS_days"])
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
                            n_jobs=5,
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

                        grid.fit(X_train.to_numpy().astype(np.float32), y_train)
                        if model_type == "cox":
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
                        else:
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
                            f"./results_reproduced/survival_functions/full_missing/{project}/{cancer}/{model_type}_{fusion}/"
                        ).mkdir(parents=True, exist_ok=True)

                        sf_df.to_csv(
                            f"./results_reproduced/survival_functions/full_missing/{project}/{cancer}/{model_type}_{fusion}/split_{outer_split}.csv",
                            index=False,
                        )


if __name__ == "__main__":
    args = parser.parse_args()
    main(
        args.fusion,
    )
