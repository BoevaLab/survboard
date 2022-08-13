import argparse
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from skorch.callbacks import EarlyStopping, LRScheduler
from skorch.dataset import Dataset
from skorch.helper import predefined_split
from sksurv.nonparametric import kaplan_meier_estimator
from torch.optim.lr_scheduler import ReduceLROnPlateau

from survboard.python.criterion import naive_neural_criterion
from survboard.python.modules import NaiveNeural
from survboard.python.skorch_nets import NaiveNeuralNet
from survboard.python.utils.get_splits import get_splits
from survboard.python.utils.utils import (
    get_blocks,
    seed_torch,
    transform_survival_target,
)

parser = argparse.ArgumentParser()
parser.add_argument("data_dir", type=str, help="Path to the folder containing data.")
parser.add_argument(
    "config_path",
    type=str,
    help="Path to the parameters needed for training in JSON format. Also includes the list of cancers available for each project.",
)
parser.add_argument("model_params", type=str, help="Path to model parameters json file")
parser.add_argument("results_path", type=str, help="Path where results should be saved")
parser.add_argument("project", type=str, help="Cancer project from which data is being used")
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
    project,
    missing_modalities,
):

    # Create paths for storing results and losses
    save_here = os.path.join(results_path, "results")
    save_loss = os.path.join(results_path, "losses")
    os.makedirs(save_here, exist_ok=True)
    os.makedirs(save_loss, exist_ok=True)

    # setup logging
    logging.basicConfig(
        handlers=[
            logging.FileHandler(os.path.join(save_here, "pancancer_experiment.log")),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger("pancancer")
    logger.setLevel(logging.DEBUG)

    # Load configuration and parameter files
    with open(config_path, "r") as f:
        config = json.load(f)

    with open(model_params, "r") as f:
        params = json.load(f)

    with open(os.path.join(save_loss, "model_params.json"), "w") as f:
        json.dump(params, f)

    # TODO: Define model to use and a name for easy access to stored results and logfiles
    # Example:
    model_name = "naive"
    setting = "pancancer"
    net = NaiveNeuralNet(
        module=NaiveNeural,
        criterion=naive_neural_criterion,
    )

    # Set seed for reproducibility
    seed_torch(params.get("random_seed"))
    logger.info(f"Starting model: {model_name}")

    assert project == "TCGA"
    # Pancancer doesn't need access to cancer names

    logger.info("Starting Experiment")

    # Load and process data with complete modalities
    data_path = "processed/pancancer_complete.csv"
    data = pd.read_csv(
        os.path.join(data_dir, data_path),
        low_memory=False,
    ).drop(columns=["patient_id"])
    time, event = data["OS_days"].astype(int), data["OS"].astype(int)
    data = data.drop(columns=["OS", "OS_days"])

    # Load and process data with incomplete modalities
    data_path_missing = "processed/pancancer_incomplete.csv"
    data_missing = pd.read_csv(
        os.path.join(data_dir, data_path_missing),
        low_memory=False,
    ).drop(columns=["patient_id"])
    time_missing, event_missing = (
        data_missing["OS_days"].astype(int),
        data_missing["OS"].astype(int),
    )
    data_missing = data_missing.drop(columns=["OS", "OS_days"])

    # Remove columns with constant values
    msk = (data != data.iloc[0]).any()
    data = data.loc[:, msk]
    data_missing = data_missing.loc[:, msk]

    # Concatenate the 2 dataframes into one
    # IMPORTANT: Do not change the order here, as the splits assume that the incomplete samples follow
    # the complete_samples.
    data_combined = pd.concat(
        [data, data_missing],
        axis=0,
    )
    time_combined = pd.concat([time, time_missing], axis=0)
    event_combined = pd.concat([event, event_missing], axis=0)

    for outer_split in range(25):
        logger.info(f"Starting split: {outer_split + 1} / 25")

        # Get split specific indices for train and test
        train_base_ix, train_ix, test_ix = get_splits(
            data_dir,
            cancer="pancancer",
            project="TCGA",
            n_samples=data_combined.shape[0],
            split_number=outer_split,
            setting="pancancer",
        )
        # Use the indices to create train and test data
        X_train = data_combined.iloc[train_ix, :]
        X_test = data_combined.iloc[test_ix, :]

        # Initialise pipeline based on columns with complete modality. We do this so we can differentiate between numerical and categorical data easily. NA values may decalre that column as having type object.
        # Numerical variables are standard scales and categorical are one hot encoded.
        ct = ColumnTransformer(
            [
                (
                    "numerical",
                    make_pipeline(StandardScaler()),
                    np.where(X_train.iloc[train_base_ix, :].dtypes != "object")[0],
                ),
                (
                    "categorical",
                    make_pipeline(
                        OneHotEncoder(sparse=False, handle_unknown="ignore"),
                        StandardScaler(),
                    ),
                    np.where(X_train.iloc[train_base_ix, :].dtypes == "object")[0],
                ),
            ]
        )

        # apply transforms to training and test data
        X_train = ct.fit_transform(X_train)
        y_train = transform_survival_target(time_combined[train_ix].values, event_combined[train_ix].values)

        stratification = X_train["clinical_cancer_type"].values

        X_train = pd.DataFrame(
            X_train,
            columns=data.columns[np.where(data.dtypes != "object")[0]].tolist()
            + [f"clinical_{i}" for i in ct.transformers_[1][1][0].get_feature_names().tolist()],
        )

        X_test = pd.DataFrame(ct.transform(X_test), columns=X_train.columns)

        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.1,
            random_state=42,
            shuffle=True,
            stratify=stratification,
        )

        valid_ds = Dataset(X_val.to_numpy().astype(np.float32), y_val.astype(str))

        # Compute parameters for skorchnet training and stopping.
        num_batches_train = (int((len(X_train) * 0.8) * 0.9)) / params.get("batch_size")
        droplast_train = (
            True if (num_batches_train - np.floor(num_batches_train)) * params.get("batch_size") == 1 else False
        )

        droplast_train = True
        num_batches_valid = np.floor((len(X_train) * 0.8) / params.get("valid_split_size")) / (
            params.get("batch_size") * 0.1
        )
        droplast_valid = (
            True if (num_batches_valid - np.floor(num_batches_valid)) * params.get("batch_size") == 1 else False
        )

        # Define parameters to use for the network following skorchnet structure.
        base_net_params = {
            "module__params": params,
            "optimizer": torch.optim.Adam,
            "max_epochs": params.get("max_epochs"),
            "lr": params.get("initial_lr"),
            "train_split": predefined_split(valid_ds),
            "batch_size": params.get("batch_size"),
            "iterator_train__drop_last": droplast_train,
            "iterator_valid__drop_last": droplast_valid,
            "iterator_train__shuffle": True,
            "module__blocks": get_blocks(X_train.columns),
            "module__p_multimodal_dropout": params.get("p_multimodal_dropout"),
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

        # Set model parameters.
        net.set_params(**base_net_params)

        # Fit model using try and except so failing models (due to numerical stability issues, etc) will produce a Kaplan-Meier estimate and not interrupt the experiment.
        try:
            net.fit(
                X_train.to_numpy().astype(np.float32),
                y_train.astype(str),
            )
            logger.info("Network Fitting Done")

            # Write out each survival function to a CSV file, as required by our webservice.
            # NOTE: Our survival functions are output with each column being a
            # time and each row being a sample. Thus, for the pancancer
            # setting we pick out the test splits belonging to each cancer in turn.
            for ix, cancer in enumerate(config["tcga_cancers"]):
                survival_functions = net.predict_survival_function(
                    X_test[X_test["clinical_cancer_type"] == cancer].to_numpy().astype(np.float32)
                )

                cancer_ix = X_train[X_train["clinical_cancer_type"] == cancer].index.values
                survival_probabilities = np.stack(
                    [i(np.unique(time_combined[cancer_ix].values)).detach().numpy() for i in survival_functions]
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
                        f"{model_name}_{setting}_complete",
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
                    f"{model_name}_{setting}_complete",
                    f"split_{outer_split}.json",
                ),
                "w",
            ) as f:
                json.dump(net.history, f)

        except Exception as e:
            logger.info(e)
            logger.info("Error encountered - replacing failing iteration with Kaplan-Meier estimate.")
            x, y = kaplan_meier_estimator(
                pd.Series(y_train).str.rsplit("|").apply(lambda x: bool(x[1])).values,
                pd.Series(y_train).str.rsplit("|").apply(lambda x: int(x[0])).values,
            )

            for ix, cancer in enumerate(config["tcga_cancers"]):
                survival_probabilities = np.stack(
                    [y for i in range(X_test[X_test["clinical_cancer_type"] == cancer].shape[0])]
                )

                sf_df = pd.DataFrame(survival_probabilities, columns=x)

                sf_df.to_csv(
                    os.path.join(
                        save_here,
                        project,
                        cancer,
                        f"{model_name}_{setting}_complete",
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
        args.project,
        args.missing_modalities,
    )
