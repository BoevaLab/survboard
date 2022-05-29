import argparse
import json
import logging
import os
import sys
from multiprocessing.sharedctypes import Value

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.utils.fixes import loguniform
from skorch.callbacks import EarlyStopping, LRScheduler
from sksurv.nonparametric import kaplan_meier_estimator
from torch.optim.lr_scheduler import ReduceLROnPlateau

from survival_benchmark.python.criterion import (
    dae_criterion,
    intermediate_fusion_mean_criterion,
    intermediate_fusion_poe_criterion,
)
from survival_benchmark.python.modules import (
    DAE,
    IntermediateFusionMean,
    IntermediateFusionPoE,
)
from survival_benchmark.python.skorch_nets import (
    DAENet,
    IntermediateFusionMeanNet,
    IntermediateFusionPoENet,
)
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
    help="How to handle missing modalities. Must be in ['impute', 'multimodal_dropout'] for DAE and IntermediateMeanFusion and in ['impute', 'multimodal_dropout', 'poe'] for IntermediateFusionPoE.",
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
    # setup logging
    logging.basicConfig(
        handlers=[
            logging.FileHandler(
                os.path.join(results_path, f"{model_name}.log")
            ),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger(f"{model_name}")
    logger.setLevel(logging.DEBUG)

    save_here = os.path.join(results_path, "results")
    if setting == "missing":
        save_here = os.path.join(save_here, missing_modalities)
    save_loss = os.path.join(results_path, "losses")
    os.makedirs(save_here, exist_ok=True)
    os.makedirs(save_loss, exist_ok=True)

    with open(config_path, "r") as f:
        config = json.load(f)

    with open(model_params, "r") as f:
        params = json.load(f)

    with open(os.path.join(save_loss, "model_params.json"), "w") as f:
        json.dump(params, f)

    seed_torch(params.get("random_seed"))
    grid_iter = params.get("grid_iter", 5)
    alpha_range = params.get("alpha_range", [0.001, 1])
    beta_range = params.get("beta_range", [0.001, 1])
    p_dropout_range = params.get("p_dropout_range", [0.0, 0.5])

    logger.info(f"Starting model: {model_name}")
    if setting == "pancancer":
        assert project == "TCGA"
        cancers = [0]  # Pancancer doesn't need access to cancer names
    else:
        cancers = config[f"{project.lower()}_cancers"]
    for cancer in cancers:
        logger.info(f"Starting cancer: {cancer}")
        if setting == "standard":
            data_path = f"processed/{project}/{cancer}_data_complete_modalities_preprocessed{'' if project.lower() == 'target' else '_fixed'}.csv"
            data = pd.read_csv(
                os.path.join(data_dir, data_path),
                low_memory=False,
            ).drop(columns=["patient_id"])
            if project == "target":
                data = data.fillna("MISSING")
            time, event = data["OS_days"].astype(int), data["OS"].astype(int)
        elif setting == "missing":
            if project != "target":
                data_path = f"processed/{project}/{cancer}_data_complete_modalities_preprocessed{'' if project.lower() == 'target' else '_fixed'}.csv"
            else:
                data_path = f"processed/{project}/{cancer}_data_complete_modalities_preprocessed.csv"
            data = pd.read_csv(
                os.path.join(data_dir, data_path),
                low_memory=False,
            ).drop(columns=["patient_id"])
            if project == "target":
                data = data.fillna("MISSING")

            if project != "target":
                data_path_missing = f"processed/{project}/{cancer}_data_incomplete_modalities_preprocessed{'' if project.lower() == 'target' else '_fixed'}.csv"
            else:
                data_path_missing = f"processed/{project}/{cancer}_data_non_complete_modalities_preprocessed.csv"
            data_missing = pd.read_csv(
                os.path.join(data_dir, data_path_missing),
                low_memory=False,
            ).drop(columns=["patient_id"])
            if project == "target":
                data_missing.iloc[
                    :,
                    [
                        i
                        for i in range(len(data_missing.columns))
                        if "clinical" in data_missing.columns[i]
                    ],
                ] = data_missing.iloc[
                    :,
                    [
                        i
                        for i in range(len(data_missing.columns))
                        if "clinical" in data_missing.columns[i]
                    ],
                ].fillna(
                    "MISSING"
                )
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
                        make_pipeline(MinMaxScaler()),
                        np.where(X_train.dtypes != "object")[0],
                    ),
                    (
                        "categorical",
                        make_pipeline(
                            OneHotEncoder(
                                sparse=False, handle_unknown="ignore"
                            )
                        ),
                        np.where(X_train.dtypes == "object")[0],
                    ),
                ]
            )
            if setting == "pancancer":
                X_train = pd.concat(
                    [X_train, data_missing],
                    axis=0,
                )
                print(X_train.shape)
                print(
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
                    ).shape
                )
                print(
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
                    ).shape
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
            else:
                for cancer in config["tcga_cancers"]:
                    X_test[cancer] = pd.DataFrame(
                        ct.transform(X_test[cancer]), columns=X_train.columns
                    )

            num_batches_train = len(X_train) / params.get("batch_size")
            droplast_train = (
                True
                if (num_batches_train - np.floor(num_batches_train))
                * params.get("batch_size")
                == 1
                else False
            )
            num_batches_valid = np.floor(
                len(X_train) / params.get("valid_split_size")
            ) / params.get("batch_size")
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
                ),
                "batch_size": params.get("batch_size"),
                "iterator_train__drop_last": droplast_train,
                "iterator_valid__drop_last": droplast_valid,
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
            }
            if model_name == "poe":
                net = IntermediateFusionPoENet(
                    module=IntermediateFusionPoE,
                    criterion=intermediate_fusion_poe_criterion,
                )
            elif model_name == "mean":
                net = IntermediateFusionMeanNet(
                    module=IntermediateFusionMean,
                    criterion=intermediate_fusion_mean_criterion,
                )
            elif model_name == "dae":
                net = DAENet(
                    module=DAE,
                    criterion=dae_criterion,
                    module__noise_factor=params.get("noise_factor"),
                )
            else:
                raise ValueError(
                    "Model name must be in ['poe', 'dae', 'mean']"
                )

            net.set_params(**base_net_params)

            # using if-else for net is the best and efficient option
            param_distributions = {
                "module__alpha": loguniform(alpha_range[0], alpha_range[1]),
                "module__p_dropout": loguniform(
                    p_dropout_range[0], p_dropout_range[1]
                ),
            }
            if model_name == "poe":
                param_distributions.update(
                    {"module__beta": loguniform(beta_range[0], beta_range[1])}
                )
            grid = RandomizedSearchCV(
                net,
                param_distributions,
                n_iter=grid_iter,
                scoring=make_scorer(
                    negative_partial_log_likelihood_loss,
                    greater_is_better=False,
                ),
                n_jobs=-1,
                refit=True,
                random_state=params.get("random_seed"),
                error_score=np.nan,
                cv=StratifiedSurvivalKFold(n_splits=5),
            )
            try:
                grid.fit(
                    X_train.to_numpy().astype(np.float32), y_train.astype(str)
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
                                pd.Series(y_train)
                                .str.rsplit("|")
                                .apply(lambda x: int(x[0]))
                                .values
                            )
                            .detach()
                            .numpy()
                            for i in survival_functions
                        ]
                    )
                    logger.info("Converting surv prob to df and saving")

                    sf_df = pd.DataFrame(
                        survival_probabilities,
                        columns=pd.Series(y_train)
                        .str.rsplit("|")
                        .apply(lambda x: int(x[0]))
                        .values,
                    )

                    sf_df.to_csv(
                        os.path.join(
                            save_here,
                            f"{project}_{cancer}_{model_name}_{outer_split}.csv",
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
                                i(time[train_splits[ix]]).detach().numpy()
                                for i in survival_functions
                            ]
                        )
                        logger.info("Converting surv prob to df and saving")

                        sf_df = pd.DataFrame(
                            survival_probabilities,
                            columns=time[train_splits[ix]],
                        )

                        sf_df.to_csv(
                            os.path.join(
                                save_here,
                                f"{project}_{cancer}_{model_name}_{outer_split}_pancancer.csv",
                            ),
                            index=False,
                        )
                        logger.info("Saving models and loss")
                with open(
                    os.path.join(
                        save_loss,
                        f"{project}_{cancer}_{model_name}_{outer_split}.json",
                    ),
                    "w",
                ) as f:
                    json.dump(grid.best_estimator_.history, f)
            except:
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
                            f"{project}_{cancer}_{model_name}_{outer_split}.csv",
                        ),
                        index=False,
                    )
                else:
                    for ix, cancer in enumerate(config["tcga_cancers"]):
                        survival_probabilities = np.stack(
                            [y for i in range(X_test.shape[0])]
                        )

                        sf_df = pd.DataFrame(survival_probabilities, columns=x)

                        sf_df.to_csv(
                            os.path.join(
                                save_here,
                                f"{project}_{cancer}_{model_name}_{outer_split}_pancancer.csv",
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
