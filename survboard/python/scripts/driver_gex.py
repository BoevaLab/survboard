import argparse
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.metrics import make_scorer
from sklearn.model_selection import (
    GridSearchCV,
    PredefinedSplit,
    train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from skorch.callbacks import EarlyStopping, LRScheduler
from skorch.dataset import Dataset
from skorch.helper import predefined_split
from sksurv.nonparametric import kaplan_meier_estimator
from survival_benchmark.python.criterion import (
    intermediate_fusion_mean_criterion,
    naive_neural_criterion,
)
from survival_benchmark.python.modules import (
    IntermediateFusionMean,
    NaiveNeural,
)
from survival_benchmark.python.skorch_nets import (
    IntermediateFusionMeanNet,
    NaiveNeuralNet,
)
from torch.optim.lr_scheduler import ReduceLROnPlateau

from survival_benchmark.python.utils.utils import (
    StratifiedSkorchSurvivalSplit,
    StratifiedSurvivalKFold,
    get_blocks,
    negative_partial_log_likelihood_loss,
    seed_torch,
    transform_survival_target,
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "data_dir", type=str, help="Path to the folder containing data."
)
parser.add_argument(
    "config_path",
    type=str,
    help="Path to the parameters needed for training in JSON format.",
)
parser.add_argument(
    "model_params", type=str, help="Path to model parameters json file"
)
parser.add_argument(
    "results_path", type=str, help="Path where results should be saved"
)
parser.add_argument(
    "model_name", type=str, help="Name of model being trained."
)
parser.add_argument(
    "project", type=str, help="Cancer project from which data is being used"
)
parser.add_argument(
    "setting",
    type=str,
    help="One of pancancer, standard, missing for the experimental setting.",
)
parser.add_argument(
    "missing_modalities",
    type=str,
    help="How to handle missing modalities. Must be in ['impute'] for NaiveNeural and in ['impute, 'multimodal_dropout'] for IntermediateFusionMean",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(
    data_dir,
    config_path,
    model_params,
    results_path,
    model_name,
    project,
    setting,
    missing_modalities,
):
    save_here = os.path.join(results_path, "results")
    save_loss = os.path.join(results_path, "losses")
    os.makedirs(save_here, exist_ok=True)
    os.makedirs(save_loss, exist_ok=True)

    # setup logging
    logging.basicConfig(
        handlers=[
            logging.FileHandler(os.path.join(save_here, f"{model_name}.log")),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger(f"{model_name}")
    logger.setLevel(logging.DEBUG)

    with open(config_path, "r") as f:
        config = json.load(f)

    with open(model_params, "r") as f:
        params = json.load(f)

    with open(os.path.join(save_loss, "model_params.json"), "w") as f:
        json.dump(params, f)

    seed_torch(params.get("random_seed"))
    logger.info(f"Starting model: {model_name}")
    if setting == "pancancer":
        assert project == "TCGA"
        cancers = [0]  # Pancancer doesn't need access to cancer names
    else:
        cancers = config[f"{project.lower()}_cancers"]
        cancers = [
            cancers[i]
            for i in range(len(cancers))
            if i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18]
        ]
    for cancer in cancers:
        logger.info(f"Starting cancer: {cancer}")
        if setting == "standard":
            data_path = f"processed/{project}/{cancer}_data_complete_modalities_preprocessed.csv"
            data = pd.read_csv(
                os.path.join(data_dir, data_path),
                low_memory=False,
            ).drop(columns=["patient_id"])

            time, event = data["OS_days"].astype(int), data["OS"].astype(int)
        elif setting == "missing":

            data_path = f"processed/{project}/{cancer}_data_complete_modalities_preprocessed.csv"
            data = pd.read_csv(
                os.path.join(data_dir, data_path),
                low_memory=False,
            ).drop(columns=["patient_id"])

            data_path_missing = f"processed/{project}/{cancer}_data_incomplete_modalities_preprocessed.csv"

            data_missing = pd.read_csv(
                os.path.join(data_dir, data_path_missing),
                low_memory=False,
            ).drop(columns=["patient_id"])

            time, event = data["OS_days"].astype(int), data["OS"].astype(int)
            time_missing, event_missing = (
                data_missing["OS_days"].astype(int),
                data_missing["OS"].astype(int),
            )
            data_missing = data_missing.drop(columns=["OS", "OS_days"])

        elif setting == "pancancer":
            data_path = "processed/pancancer_complete.csv"
            data = pd.read_csv(
                os.path.join(data_dir, data_path),
                low_memory=False,
            ).drop(columns=["patient_id"])
            data_path_missing = "processed/pancancer_incomplete.csv"
            data_missing = pd.read_csv(
                os.path.join(data_dir, data_path_missing),
                low_memory=False,
            ).drop(columns=["patient_id"])
            time, event = data["OS_days"].astype(int), data["OS"].astype(int)
            time_missing, event_missing = (
                data_missing["OS_days"].astype(int),
                data_missing["OS"].astype(int),
            )
            data_missing = data_missing.drop(columns=["OS", "OS_days"])
        else:
            raise ValueError(
                "`setting` must be in ['standard', 'missing', 'pancancer']"
            )

        data = data.drop(columns=["OS", "OS_days"])
        msk = (data != data.iloc[0]).any()
        data = data.loc[:, msk]
        if setting != "standard":
            data_missing = data_missing.loc[:, msk]
        if setting != "pancancer":
            # Train Test Splits
            train_splits = pd.read_csv(
                os.path.join(
                    data_dir, f"splits/{project}/{cancer}_train_splits.csv"
                )
            )
            test_splits = pd.read_csv(
                os.path.join(
                    data_dir, f"splits/{project}/{cancer}_test_splits.csv"
                )
            )
        else:
            train_splits = {}
            test_splits = {}
            for cancer in config["tcga_cancers"]:
                train_splits[cancer] = pd.read_csv(
                    os.path.join(
                        data_dir, f"splits/{project}/{cancer}_train_splits.csv"
                    )
                )
                test_splits[cancer] = pd.read_csv(
                    os.path.join(
                        data_dir, f"splits/{project}/{cancer}_test_splits.csv"
                    )
                )

        for outer_split in range(25):
            logger.info(f"Starting split: {outer_split + 1} / 25")

            if setting != "pancancer":
                train_ix = (
                    train_splits.iloc[outer_split, :]
                    .dropna()
                    .values.astype(int)
                )
                test_ix = (
                    test_splits.iloc[outer_split, :]
                    .dropna()
                    .values.astype(int)
                )
            else:
                train_ix = []
                test_ix = []
                for cancer in train_splits.keys():
                    train_ix.append(
                        data.loc[data["clinical_cancer_type"] == cancer, :]
                        .iloc[
                            train_splits[cancer]
                            .iloc[outer_split, :]
                            .dropna()
                            .values,
                            :,
                        ]
                        .index.values.astype(int)
                    )
                    test_ix.append(
                        data.loc[data["clinical_cancer_type"] == cancer, :]
                        .iloc[
                            test_splits[cancer]
                            .iloc[outer_split, :]
                            .dropna()
                            .values,
                            :,
                        ]
                        .index.values.astype(int)
                    )

            if setting != "pancancer":
                X_test = data.iloc[test_ix, :]
                X_train = data.iloc[train_ix, :]
            else:
                X_train = data.iloc[
                    np.array(
                        [item for sublist in train_ix for item in sublist]
                    ),
                    :,
                ]
                X_test = {}
                for cancer in config["tcga_cancers"]:
                    X_test[cancer] = data.loc[
                        data["clinical_cancer_type"] == cancer
                    ].iloc[
                        test_splits[cancer]
                        .iloc[outer_split, :]
                        .dropna()
                        .values,
                        :,
                    ]

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
            if setting == "pancancer":
                X_train = pd.concat(
                    [
                        X_train,
                        data_missing,
                    ],
                    axis=0,
                )
                y_train = transform_survival_target(
                    pd.concat(
                        [
                            time[
                                np.array(
                                    [
                                        item
                                        for sublist in train_ix
                                        for item in sublist
                                    ]
                                )
                            ],
                            time_missing,
                        ],
                        axis=0,
                    ).to_numpy(),
                    pd.concat(
                        [
                            event[
                                np.array(
                                    [
                                        item
                                        for sublist in train_ix
                                        for item in sublist
                                    ]
                                )
                            ],
                            event_missing,
                        ],
                        axis=0,
                    ).to_numpy(),
                )
            elif setting == "missing":
                X_train = pd.concat(
                    [X_train, data_missing],
                    axis=0,
                )
                y_train = transform_survival_target(
                    np.append(time[train_ix].values, time_missing.values),
                    np.append(event[train_ix].values, event_missing),
                )
            else:
                y_train = transform_survival_target(
                    time[train_ix].values, event[train_ix].values
                )
            if setting == "pancancer":
                stratification = X_train["clinical_cancer_type"].values
            X_train = ct.fit_transform(X_train)

            X_train = pd.DataFrame(
                X_train,
                columns=data.columns[
                    np.where(data.dtypes != "object")[0]
                ].tolist()
                + [
                    f"clinical_{i}"
                    for i in ct.transformers_[1][1][0]
                    .get_feature_names()
                    .tolist()
                ],
            )
            if setting != "pancancer":
                X_test = pd.DataFrame(
                    ct.transform(X_test), columns=X_train.columns
                )
                X_train = [[i for i in X_train.columns if i.rsplit("_")[0] in ["gex"]]]
                X_test = [[i for i in X_test.columns if i.rsplit("_")[0] in ["gex"]]]
            else:
                for cancer in config["tcga_cancers"]:
                    X_test[cancer] = pd.DataFrame(
                        ct.transform(X_test[cancer]), columns=X_train.columns
                    )
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train,
                    y_train,
                    test_size=0.1,
                    random_state=42,
                    shuffle=True,
                    stratify=stratification,
                )

                valid_ds = Dataset(
                    X_val.to_numpy().astype(np.float32), y_val.astype(str)
                )
                X_train_cv, X_val_cv, y_train_cv, y_val_cv = train_test_split(
                    X_train,
                    y_train,
                    test_size=0.1,
                    random_state=42,
                    shuffle=True,
                    stratify=stratification[X_train.index],
                )
                cv = PredefinedSplit(
                    [
                        -1 if i not in X_val_cv.index else 1
                        for i in range(X_train.shape[0])
                    ]
                )

            num_batches_train = (int((len(X_train) * 0.8) * 0.9)) / params.get(
                "batch_size"
            )
            num_batches_train = (np.round(len(X_train) * 0.9)) / params.get(
                "batch_size"
            )
            droplast_train = (
                True
                if (num_batches_train - np.floor(num_batches_train))
                * params.get("batch_size")
                == 1
                else False
            )
            if setting == "pancancer":
                droplast_train = True
            num_batches_valid = np.floor(
                (len(X_train) * 0.8) / params.get("valid_split_size")
            ) / (params.get("batch_size") * 0.1)
            droplast_valid = (
                True
                if (num_batches_valid - np.floor(num_batches_valid))
                * params.get("batch_size")
                == 1
                else False
            )
            base_net_params = {
                "module__params": params,
                "optimizer": torch.optim.Adam,
                "max_epochs": params.get("max_epochs"),
                "lr": params.get("initial_lr"),
                "train_split": StratifiedSkorchSurvivalSplit(
                    params.get("valid_split_size"), stratified=True
                )
                if setting != "pancancer"
                else predefined_split(valid_ds),
                "batch_size": params.get("batch_size"),
                "iterator_train__drop_last": droplast_train,
                "iterator_valid__drop_last": droplast_valid,
                "iterator_train__shuffle": True,
                "module__blocks": get_blocks(X_train.columns),
                "module__p_multimodal_dropout": params.get(
                    "p_multimodal_dropout"
                ),
                "module__missing_modalities": missing_modalities,
                "callbacks": [
                    (
                        "sched",
                        LRScheduler(
                            ReduceLROnPlateau,
                            monitor="valid_loss",
                            patience=params.get("schedule_patience"),
                        ),
                    ),
                    (
                        "es",
                        EarlyStopping(
                            monitor="valid_loss",
                            patience=params.get("es_patience"),
                            load_best=True,
                        ),
                    ),
                ],
                "verbose": False,
            }
            if model_name == "mean":
                net = IntermediateFusionMeanNet(
                    module=IntermediateFusionMean,
                    criterion=intermediate_fusion_mean_criterion,
                )
            elif model_name == "naive":
                net = NaiveNeuralNet(
                    module=NaiveNeural,
                    criterion=naive_neural_criterion,
                )
            else:
                raise ValueError("Model name must be in ['mean', 'naive']")

            net.set_params(**base_net_params)
            param_distributions = {
                "module__p_dropout": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
            }
            if setting != "pancancer":
                cv = StratifiedSurvivalKFold(n_splits=5)

            grid = GridSearchCV(
                net,
                param_distributions,
                scoring=make_scorer(
                    negative_partial_log_likelihood_loss,
                    greater_is_better=False,
                ),
                n_jobs=-1,
                refit=True,
                cv=cv,
            )
            try:
                grid.fit(
                    X_train.to_numpy().astype(np.float32),
                    y_train.astype(str),
                )
                logger.info("Network Fitting Done")
                if setting != "pancancer":
                    survival_functions = (
                        grid.best_estimator_.predict_survival_function(
                            X_test.to_numpy().astype(np.float32)
                        )
                    )
                    survival_probabilities = np.stack(
                        [
                            i(
                                np.unique(
                                    pd.Series(y_train)
                                    .str.rsplit("|")
                                    .apply(lambda x: int(x[0]))
                                    .values
                                )
                            )
                            .detach()
                            .numpy()
                            for i in survival_functions
                        ]
                    )
                    logger.info("Converting surv prob to df and saving")

                    sf_df = pd.DataFrame(
                        survival_probabilities,
                        columns=np.unique(
                            pd.Series(y_train)
                            .str.rsplit("|")
                            .apply(lambda x: int(x[0]))
                            .values
                        ),
                    )

                    sf_df.to_csv(
                        os.path.join(
                            save_here,
                            project,
                            cancer,
                            f"{model_name}_{setting}_gex_only",
                            f"split_{outer_split}.csv",
                        ),
                        index=False,
                    )
                    logger.info("Saving models and loss")
                else:
                    for ix, cancer in enumerate(config["tcga_cancers"]):
                        survival_functions = (
                            grid.best_estimator_.predict_survival_function(
                                X_test[cancer].to_numpy().astype(np.float32)
                            )
                        )
                        survival_probabilities = np.stack(
                            [
                                i(np.unique(time[train_ix[ix]].values))
                                .detach()
                                .numpy()
                                for i in survival_functions
                            ]
                        )
                        logger.info("Converting surv prob to df and saving")

                        sf_df = pd.DataFrame(
                            survival_probabilities,
                            columns=np.unique(time[train_ix[ix]].values),
                        )

                        sf_df.to_csv(
                            os.path.join(
                                save_here,
                                project,
                                cancer,
                                f"{model_name}_{setting}_gex_only",
                                f"split_{outer_split}.csv",
                            ),
                            index=False,
                        )
                        logger.info("Saving models and loss")
                with open(
                    os.path.join(
                        save_loss,
                        project,
                        cancer,
                        f"{model_name}_{setting}_gex_only",
                        f"split_{outer_split}.json",
                    ),
                    "w",
                ) as f:
                    json.dump(grid.best_estimator_.history, f)
            except Exception as e:
                logger.info(e)
                logger.info(
                    "Error encountered - replacing failing iteration with Kaplan-Meier estimate."
                )
                x, y = kaplan_meier_estimator(
                    pd.Series(y_train)
                    .str.rsplit("|")
                    .apply(lambda x: bool(x[1]))
                    .values,
                    pd.Series(y_train)
                    .str.rsplit("|")
                    .apply(lambda x: int(x[0]))
                    .values,
                )
                if setting != "pancancer":
                    survival_probabilities = np.stack(
                        [y for i in range(X_test.shape[0])]
                    )

                    sf_df = pd.DataFrame(survival_probabilities, columns=x)

                    sf_df.to_csv(
                        os.path.join(
                            save_here,
                            project,
                            cancer,
                            f"{model_name}_{setting}_gex_only",
                            f"split_{outer_split}.csv",
                        ),
                        index=False,
                    )
                else:
                    for ix, cancer in enumerate(config["tcga_cancers"]):
                        survival_probabilities = np.stack(
                            [y for i in range(X_test[cancer].shape[0])]
                        )

                        sf_df = pd.DataFrame(survival_probabilities, columns=x)

                        sf_df.to_csv(
                            os.path.join(
                                save_here,
                                project,
                                cancer,
                                f"{model_name}_{setting}_gex_only",
                                f"split_{outer_split}.csv",
                            ),
                            index=False,
                        )
    logger.info("Experiment Complete")


if __name__ == "__main__":
    args = parser.parse_args()
    main(
        args.data_dir,
        args.config_path,
        args.model_params,
        args.results_path,
        args.model_name,
        args.project,
        args.setting,
        args.missing_modalities,
    )
