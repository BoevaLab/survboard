import argparse
import json
import logging
import os
import sys

from sklearn.model_selection import RandomSearchCV

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    Normalizer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.utils.fixes import loguniform
from skorch.callbacks import EarlyStopping, LRScheduler
from skorch.dataset import ValidSplit

from survival_benchmark.python.modules.MultiSurv.dataset_benchmark import (
    MultimodalDataset,
)
from survival_benchmark.python.modules.MultiSurv.loss import Loss
from survival_benchmark.python.modules.MultiSurv.multisurv import (
    MultiSurv,
    MultiSurvModel,
)
from survival_benchmark.python.utils.utils import (
    FixRandomSeed,
    StratifiedSkorchSurvivalSplit,
    StratifiedSurvivalKFold,
    cox_criterion,
    get_blocks,
    seed_torch,
    transform_survival_target,
)

# from skorch.net import NeuralNet
# from hyperband import HyperbandSearchCV
# from scipy.stats import uniform

# from sklearn.metrics import make_scorer


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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(
    data_dir, config_path, model_params, results_path, model_name, project
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

    seed_torch()

    save_here = os.path.join(results_path, "results")
    save_model = os.path.join(results_path, "models")
    save_loss = os.path.join(results_path, "losses")
    os.makedirs(save_here, exist_ok=True)
    os.makedirs(save_loss, exist_ok=True)
    os.makedirs(save_model, exist_ok=True)

    with open(config_path, "r") as f:
        config = json.load(f)

    with open(model_params, "r") as f:
        params = json.load(f)

    with open(os.path.join(save_model, "model_params.json"), "w") as f:
        json.dump(params, f)
    logger.info(f"Starting model: {model_name}")

    for cancer in config[f"{project.lower()}_cancers"]:
        # for cancer in ["SKCM"]:
        logger.info(f"Starting cancer: {cancer}")
        input_size = {}
        data_path = f"processed/{project}/{cancer}_data_complete_modalities_preprocessed{'' if project.tolower() == 'target' else '_fixed'}.csv"
        data = pd.read_csv(
            os.path.join(data_dir, data_path), index_col="patient_id"
        )
        if missing:
            data_path = f"processed/{project}/{cancer}_data_incomplete_modalities_preprocessed{'' if project.tolower() == 'target' else '_fixed'}.csv"
            data_missing = pd.read_csv(
                os.path.join(data_dir, data_path), index_col="patient_id"
            )
        time, event = data["OS_days"], data["OS"]
        data = data.drop(columns=["OS", "OS_days"])

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

        for outer_split in range(train_splits.shape[0]):
            logger.info(f"Starting split: {outer_split + 1} / 25")

            train_ix = train_splits.iloc[outer_split, :].dropna().values
            test_ix = test_splits.iloc[outer_split, :].dropna().values

            X_train = data.iloc[train_ix, :]
            X_test = data.iloc[test_ix, :]
            y_train = transform_survival_target(
                time[train_ix], event[train_ix]
            )
            y_test = transform_survival_target(time[test_ix], event[test_ix])

            ct = ColumnTransformer(
                [
                    (
                        "numerical",
                        VarianceThreshold(),
                        StandardScaler(),
                        np.where(X_train.dtypes != "object")[0],
                    ),
                    (
                        "categorical",
                        make_pipeline(
                            OneHotEncoder(
                                sparse=False, handle_unknown="ignore"
                            ),
                            VarianceThreshold(),
                            StandardScaler(),
                        ),
                        np.where(X_train.dtypes == "object")[0],
                    ),
                ]
            )
            X_train = ct.fit_transform(X_train)
            X_train = pd.DataFrame(
                X_train,
                columns=X_train.columns[
                    np.where(X_train.dtypes != "object")[0]
                ].tolist()
                + [
                    f"clinical_{i}"
                    for i in ct.transformers_[1][1][0]
                    .get_feature_names()
                    .tolist()
                ],
            )
            X_test = pd.DataFrame(
                ct.transform(X_test), columns=X_train.columns
            )
            if missing:
                X_train = pd.concat(
                    [
                        X_train,
                        pd.DataFrame(
                            ct.transform(data_missing), columns=X_train.columns
                        ),
                    ],
                    axis=1,
                )

            net = np.nan
            grid = RandomSearchCV(
                net,
                param_distribution,  # TODO
                n_iter=n_iter,  # TODO
                scoring=make_scorer(negative_partial_log_likelihood_loss),
                n_jobs=1,
                refit=True,
                random_state=42,
                error_score=np.nan,
                cv=StratifiedSurvivalKFold(n_splits=5),  # TODO
            )
            net.fit(X_train, y_train)

            logger.info("Network Fitting Done")
            survival_functions = net.predict_survival_function(X_test)
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
                )
            )
            logger.info("Saving models and loss")
            with open(
                os.path.join(
                    save_loss,
                    f"{project}_{cancer}_{model_name}_{outer_split}.json",
                ),
                "w",
            ) as f:
                json.dump(net.history, f)

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
    )
