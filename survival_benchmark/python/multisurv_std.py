import os
import sys
import argparse
import logging
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import plotly

# from skorch.net import NeuralNet
# from hyperband import HyperbandSearchCV
# from scipy.stats import uniform

# from sklearn.metrics import make_scorer

# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.utils.fixes import loguniform
from skorch.callbacks import EarlyStopping, LRScheduler
from skorch.dataset import ValidSplit
from survival_benchmark.python.modules.MultiSurv.dataset_benchmark import MultimodalDataset
from survival_benchmark.python.modules.MultiSurv.multisurv import MultiSurv, MultiSurvModel
from survival_benchmark.python.modules.MultiSurv.loss import Loss
from survival_benchmark.python.utils.utils import (
    FixRandomSeed,
    StratifiedSkorchSurvivalSplit,
    StratifiedSurvivalKFold,
    cox_criterion,
    get_blocks,
    seed_torch,
)

parser = argparse.ArgumentParser()
parser.add_argument("data_dir", type=str, help="Path to the folder containing data.")
parser.add_argument("config_path", type=str, help="Path to the parameters needed for training in JSON format.")
parser.add_argument("model_params", type=str, help="Path to model parameters json file")
parser.add_argument("results_path", type=str, help="Path where results should be saved")
parser.add_argument("model_name", type=str, help="Name of model being trained.")
parser.add_argument("project", type=str, help="Cancer project from which data is being used")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(data_dir, config_path, model_params, results_path, model_name, project):
    # setup logging
    logging.basicConfig(
        handlers=[
            logging.FileHandler(os.path.join(results_path, f"{model_name}.log")),
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

        data_path = f"processed/{project}/{cancer}_data_complete_modalities_preprocessed.csv"

        idx_col = "clinical_patient_id" if project == "TCGA" else "patient_id"
        data = pd.read_csv(os.path.join(data_dir, data_path), index_col=idx_col)

        # TODO/INFO: Be careful here wrt missing modality case
        data = data.fillna("NA")

        train_splits = pd.read_csv(os.path.join(data_dir, f"splits/{project}/{cancer}_train_splits.csv"))
        test_splits = pd.read_csv(os.path.join(data_dir, f"splits/{project}/{cancer}_test_splits.csv"))

        for outer_split in range(train_splits.shape[0]):
            logger.info(f"Starting split: {outer_split + 1} / 25")

            train_ix = train_splits.iloc[outer_split, :].dropna().values
            test_ix = test_splits.iloc[outer_split, :].dropna().values

            X_train = data.iloc[train_ix, :]
            y_train = X_train["OS"]
            X_test = data.iloc[test_ix, :]

            msk = (X_train != X_train.iloc[0]).any()
            X_train = X_train.loc[:, msk]
            X_test = X_test.loc[:, msk]

            train_dataset = MultimodalDataset(X_train)
            test_dataset = MultimodalDataset(
                X_test,
                categorical_encoder=train_dataset.cat_encoder,
                cnv_encoder=train_dataset.cnv_encoder,
                scaler_test=train_dataset.scaler,
                mode="test",
            )

            net = MultiSurvModel(
                module=MultiSurv,
                criterion=Loss,
                optimizer=torch.optim.Adam,
                module__data_modalities=train_dataset.input_size,
                module__output_intervals=params.get("output_intervals", torch.arange(0, 21, 1)),
                train_split=ValidSplit(10, stratified=True),
                criterion__aux_criterion=None,
                criterion__is_multimodal=len(train_dataset.input_size) > 1,
                max_epochs=params.get("max_epochs", 100),
                batch_size=params.get("batch_size", 128),
                verbose=1,
                callbacks=[
                    ("seed", FixRandomSeed(config["seed"])),
                    (
                        "sched",
                        LRScheduler(
                            torch.optim.lr_scheduler.ReduceLROnPlateau,
                            monitor="valid_loss",
                            patience=params.get("scheduler_patience", 10),
                            factor=params.get("scheduler_factor", 0.1),
                        ),
                    ),
                    (
                        "es",
                        EarlyStopping(
                            patience=params.get("ES_patience", 25),
                            monitor="valid_loss",
                            load_best=True,
                        ),
                    ),
                ],
                device=device,
            )

            logger.info("Starting LR Finder")
            # # LR range
            net.initialize()
            lr_bs = 64
            num_batches = len(data) / lr_bs
            droplast = True if (num_batches - np.floor(num_batches)) * lr_bs > 1 else False
            best_lr = net.test_lr_range(
                dataloader=torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=lr_bs,
                    drop_last=droplast,
                ),
                optimizer=net.optimizer(net.module_.parameters(), lr=1e-4),
                criterion=net.criterion_,
                auxiliary_criterion=None,
                output_intervals=torch.arange(0, 21, 1),
                model=net.module_,
                init_value=1e-6,
                final_value=10,
            )

            logger.info(f"Best LR Found - {best_lr}")
            logger.info("Set params for network")
            net.set_params(**{"lr": best_lr})
            logger.info("Fitting Network")
            net.fit(train_dataset, y_train)
            logger.info("Network Fitting Done")
            survival_prob = net.predict_survival_function(test_dataset)
            logger.info("Converting surv prob to df and saving")
            sf_df = pd.DataFrame(survival_prob, columns=net.module__output_intervals)
            sf_df.to_csv(
                os.path.join(
                    save_here,
                    f"{project}_{cancer}_{model_name}_{outer_split}.csv",
                )
            )
            logger.info("Saving models and loss")
            # save models and loss
            net.save_params(f_params=os.path.join(save_model, f"{project}_{cancer}_{model_name}_{outer_split}.pkl"))
            with open(os.path.join(save_loss, f"{project}_{cancer}_{model_name}_{outer_split}.json"), "w") as f:
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
