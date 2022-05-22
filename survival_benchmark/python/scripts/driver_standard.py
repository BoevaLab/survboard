import json
import sys

import numpy as np
import pandas as pd
import torch
from hyperband import HyperbandSearchCV
from scipy.stats import uniform
from sklearn.compose import ColumnTransformer
from sklearn.metrics import make_scorer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.fixes import loguniform
from skorch.callbacks import EarlyStopping, LRScheduler

from survival_benchmark.python.modules.modules import (
    GDP,
    CoxPHNet,
    DeepSurv,
    GDPNet,
)
from survival_benchmark.python.utils.utils import (
    FixRandomSeed,
    StratifiedSkorchSurvivalSplit,
    StratifiedSurvivalKFold,
    cox_criterion,
    get_blocks,
    negative_partial_log_likelihood_loss,
    seed_torch,
    transform_survival_target,
)

model_mapping = ["deepsurv", "gdp"]

param_spaces = [
    {
        "lr": loguniform(0.01, 0.0001),
        "module__p_dropout": uniform(0.0, 0.5),
        "optimizer__weight_decay": loguniform(0.01, 0.0001),
        "batch_size": [64, 128, 256],
    },
    {
        "lr": loguniform(0.01, 0.0001),
        "module__p_dropout": uniform(0.0, 0.5),
        "module__alpha": uniform(0.0, 1.0),
        "module__scale": loguniform(0.01, 0.0001),
        "batch_size": [64, 128, 256],
    },
]


def main():
    seed_torch()
    with open("config/config.json") as f:
        config = json.load(f)
    cox_loss_scorer = make_scorer(
        score_func=negative_partial_log_likelihood_loss,
        greater_is_better=False,
    )
    for project in ["TCGA", "TARGET", "ICGC"]:
        for ix in range(2):
            print(f"Starting model: {model_mapping[ix]}")
            for cancer in config[f"{project.tolower()}_cancers"]:
                print(f"Starting cancer: {cancer}")
                data = pd.read_csv(
                    f"~/boeva_lab_scratch/data/projects/David/Nikita_David_survival_benchmark/survival_benchmark/data/processed/{project}/{cancer}_data_complete_modalities_preprocessed.csv"
                )
                time, event = data["OS_days"].values, data["OS"].values

                train_splits = pd.read_csv(
                    f"~/boeva_lab_scratch/data/projects/David/Nikita_David_survival_benchmark/survival_benchmark/data/splits/{project}/{cancer}_train_splits.csv"
                )
                test_splits = pd.read_csv(
                    f"~/boeva_lab_scratch/data/projects/David/Nikita_David_survival_benchmark/survival_benchmark/data/splits/{project}/{cancer}_test_splits.csv"
                )
                y = transform_survival_target(time, event)
                data = data[
                    [
                        i
                        for i in data.columns
                        if i
                        not in [
                            "OS",
                            "OS_days",
                            "patient_id",
                            "clinical_patient_id",
                        ]  # I slightly fucked up the naming for TCGA, so patient_id there also has a clinical
                    ]
                ].fillna("NA")
                ct = ColumnTransformer(
                    [
                        (
                            "numerical",
                            StandardScaler(),
                            np.where(data.dtypes != "object")[0],
                        ),
                        (
                            "categorical",
                            make_pipeline(
                                OneHotEncoder(
                                    sparse=False, handle_unknown="ignore"
                                ),
                                StandardScaler(),
                            ),
                            np.where(data.dtypes == "object")[0],
                        ),
                    ]
                )
                for outer_split in range(train_splits.shape[0]):
                    print(f"Starting split: {outer_split + 1} / 25")
                    train_ix = (
                        train_splits.iloc[outer_split, :].dropna().values
                    )
                    test_ix = test_splits.iloc[outer_split, :].dropna().values
                    X_train = ct.fit_transform(data.iloc[train_ix, :])

                    X_train = pd.DataFrame(
                        X_train,
                        columns=data.columns[
                            np.where(data.dtypes != "object")[0]
                        ].tolist()
                        + [
                            f"clinical_{i.rsplit('_')[1]}"
                            for i in ct.transformers_[1][1][0]
                            .get_feature_names()
                            .tolist()
                        ],
                    )
                    X_test = pd.DataFrame(
                        ct.transform(data.iloc[test_ix, :]),
                        columns=X_train.columns,
                    )

                    msk = (X_train != X_train.iloc[0]).any()
                    X_train = X_train.loc[:, msk]
                    X_test = X_test.loc[:, msk]
                    if ix == 0:
                        net = CoxPHNet(
                            module=DeepSurv,
                            module__input_dimension=X_train.shape[1],
                            module__hidden_layer_sizes=[
                                200,
                                100,
                            ],  # As in the paper
                            criterion=cox_criterion,
                            train_split=StratifiedSkorchSurvivalSplit(
                                5, stratified=True
                            ),
                            verbose=0,
                            optimizer=torch.optim.AdamW,
                            callbacks=[
                                ("seed", FixRandomSeed(config["seed"])),
                                (
                                    "sched",
                                    LRScheduler(
                                        torch.optim.lr_scheduler.ReduceLROnPlateau,
                                        monitor="valid_loss",
                                    ),
                                ),
                            ],
                        )
                    elif ix == 1:
                        net = GDPNet(
                            module=GDP,
                            module__hidden_layer_sizes=[200, 100],
                            module__blocks=get_blocks(X_train.columns),
                            criterion=cox_criterion,
                            train_split=StratifiedSkorchSurvivalSplit(
                                5, stratified=True
                            ),
                            verbose=0,
                            optimizer=torch.optim.Adam,
                            callbacks=[
                                ("seed", FixRandomSeed(config["seed"])),
                                (
                                    "sched",
                                    LRScheduler(
                                        torch.optim.lr_scheduler.ReduceLROnPlateau,
                                        monitor="valid_loss",
                                    ),
                                ),
                            ],
                        )
                    grid = HyperbandSearchCV(
                        estimator=net,
                        param_distributions=param_spaces[ix],
                        resource_param="max_epochs",
                        scoring=cox_loss_scorer,
                        cv=StratifiedSurvivalKFold(),
                        random_state=config["seed"],
                        refit=False,
                        max_iter=81,
                        n_jobs=1,  # TODO: Change if multiple GPUs?
                    )
                    grid.fit(
                        X_train.to_numpy().astype(np.float32),
                        y[train_ix.astype(int)].astype(str),
                    )
                    if ix == 0:
                        net = CoxPHNet(
                            module=DeepSurv,
                            module__input_dimension=X_train.shape[1],
                            module__hidden_layer_sizes=[
                                200,
                                100,
                            ],  # As in the paper
                            criterion=cox_criterion,
                            max_epochs=100,
                            train_split=StratifiedSkorchSurvivalSplit(
                                5, stratified=True
                            ),
                            verbose=1,
                            callbacks=[
                                ("seed", FixRandomSeed(config["seed"])),
                                (
                                    "sched",
                                    LRScheduler(
                                        torch.optim.lr_scheduler.ReduceLROnPlateau,
                                        monitor="valid_loss",
                                    ),
                                ),
                                (
                                    "es",
                                    EarlyStopping(
                                        patience=10,
                                        monitor="valid_loss",
                                        load_best=True,
                                    ),
                                ),
                            ],
                        )
                    elif ix == 1:
                        net = GDPNet(
                            module=GDP,
                            module__hidden_layer_sizes=[200, 100],
                            module__blocks=get_blocks(X_train.columns),
                            criterion=cox_criterion,
                            max_epochs=100,
                            train_split=StratifiedSkorchSurvivalSplit(
                                5, stratified=True
                            ),
                            verbose=1,
                            callbacks=[
                                ("seed", FixRandomSeed(config["seed"])),
                                (
                                    "sched",
                                    LRScheduler(
                                        torch.optim.lr_scheduler.ReduceLROnPlateau,
                                        monitor="valid_loss",
                                    ),
                                ),
                                (
                                    "es",
                                    EarlyStopping(
                                        patience=10,
                                        monitor="valid_loss",
                                        load_best=True,
                                    ),
                                ),
                            ],
                        )
                    net.set_params(
                        **{
                            key: val
                            for key, val in grid.best_params_.items()
                            if key != "max_epochs"
                        }
                    )
                    net.fit(
                        X_train.to_numpy().astype(np.float32),
                        y[train_ix.astype(int)].astype(str),
                    )
                    survival_functions = net.predict_survival_function(
                        X_test.to_numpy().astype(np.float32)
                    )

                    pd.DataFrame(
                        [
                            i(time[train_ix.astype(int)]).detach().numpy()
                            for i in survival_functions
                        ],
                        columns=time[train_ix.astype(int)].astype(int),
                    ).to_csv(
                        f"./data/results/{project}_{cancer}_{model_mapping[ix]}_{outer_split}.csv",
                        index=False,
                    )


if __name__ == "__main__":
    sys.exit(main())
