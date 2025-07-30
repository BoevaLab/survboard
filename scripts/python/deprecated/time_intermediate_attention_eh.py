import argparse
import json
import os
import pathlib
import sys

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
    transform_discrete_time,
)

with open(snakemake.log[0], "w") as f:
    sys.stderr = f
    sys.stdout = f
    with open(os.path.join("./config/", "config.json"), "r") as f:
        config = json.load(f)
    g = np.random.default_rng(config.get("random_state"))

    for fusion in ["intermediate_attention"]:
        for model_type in ["eh"]:
            for project in ["TCGA"]:
                for cancer in [snakemake.wildcards["cancer"]]:
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
                            if data.columns[i].rsplit("_")[0]
                            in ["clinical", "gex", "OS"]
                        ],
                    ]
                    data_helper = data.copy(deep=True).drop(columns=["OS_days", "OS"])
                    X_train = data.sort_values(
                        by="OS_days", ascending=True
                    ).reset_index(drop=True)
                    time_train = X_train["OS_days"].values
                    event_train = X_train["OS"].values
                    X_train = X_train.drop(columns=["OS", "OS_days"])
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
                    grid.fit(X_train.to_numpy().astype(np.float32), y_train)
                    pathlib.Path(
                        f"results_reproduced/timings/intermediate_attention_eh_{snakemake.wildcards['cancer']}"
                    ).touch()
