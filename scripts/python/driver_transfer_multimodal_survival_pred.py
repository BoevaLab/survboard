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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sksurv.linear_model.coxph import BreslowEstimator

from survboard.python.model.model import SKORCH_MODULE_FACTORY
from survboard.python.model.multimodal_survival_pred import (
    Model,
    MyDataset,
    compose_run_tag,
    get_dataloaders,
    setup_seed,
)
from survboard.python.utils.misc_utils import seed_torch

modality_remapping = {
    "clinical": "clinical",
    "meth": "meth",
    "gex": "mRNA",
    "mirna": "miRNA",
    "mut": "mut",
    "rppa": "rppa",
    "cnv": "CNV",
}


def filter_modality(data, quantile):
    data_var = data.apply(np.var, axis=0)
    data_var_quantile = np.quantile(data_var, quantile)
    chosen_cols = data.iloc[:, np.where(data_var > data_var_quantile)[0]].columns
    return chosen_cols


def main():
    with open(os.path.join("./config/", "config.json"), "r") as f:
        config = json.load(f)

    g = np.random.default_rng(config.get("random_state"))
    seed_torch(config.get("random_state"))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    setup_seed(config.get("random_state"))
    for fusion in ["multimodal_survival_pred"]:
        for model_type in ["multimodal_survival_pred"]:
            for project in ["validation"]:
                for cancer in ["PAAD", "LIHC"]:

                    if cancer == "PAAD":
                        split_project = "ICGC"
                        split_cancer = "PACA-CA"
                        train_data_full_path = (
                            f"./data_reproduced/validation/{cancer}/icgc_paca_ca.csv"
                        )
                        predict_data_path = (
                            f"./data_reproduced/validation/{cancer}/tcga_paad.csv"
                        )
                    else:
                        split_project = "TCGA"
                        split_cancer = "LIHC"
                        train_data_full_path = (
                            f"./data_reproduced/validation/{cancer}/tcga_lihc.csv"
                        )
                        predict_data_path = (
                            f"./data_reproduced/validation/{cancer}/icgc_liri_jp.csv"
                        )

                    train_data_full = pd.read_csv(
                        os.path.join(train_data_full_path),
                        low_memory=False,
                    )
                    predict_data = pd.read_csv(
                        os.path.join(predict_data_path),
                        low_memory=False,
                    )
                    feature_names = train_data_full.columns
                    column_types = (
                        pd.Series(feature_names)
                        .str.rsplit("_")
                        .apply(lambda x: x[0])
                        .values
                    )

                    survival_data = train_data_full.iloc[
                        :, np.where(column_types == "OS")[0]
                    ]
                    train_splits = pd.read_csv(
                        os.path.join(
                            f"./data_reproduced/splits/{split_project}/{split_cancer}_train_splits.csv"
                        )
                    )

                    available_modalities = [
                        i for i in np.unique(column_types) if i != "OS"
                    ]
                    clinical_cols = train_data_full.iloc[
                        :, np.where(np.isin(column_types, ["OS", "clinical"]))[0]
                    ].columns.tolist()
                    chosen_cols = []
                    for modality in [
                        i for i in available_modalities if i != "clinical"
                    ]:
                        if modality in ["gex", "meth", "mut", "cnv"]:
                            chosen_cols += filter_modality(
                                train_data_full.iloc[
                                    :, np.where(np.isin(column_types, [modality]))[0]
                                ],
                                0.9,
                            ).tolist()
                        else:
                            chosen_cols += filter_modality(
                                train_data_full.iloc[
                                    :, np.where(np.isin(column_types, [modality]))[0]
                                ],
                                0.0,
                            ).tolist()

                    chosen_cols = clinical_cols + chosen_cols

                    data_overall = train_data_full[chosen_cols]
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
                                make_pipeline(MinMaxScaler()),
                                np.where(data_overall_vars.dtypes != "object")[0],
                            ),
                            (
                                "categorical",
                                make_pipeline(
                                    OneHotEncoder(
                                        sparse=False, handle_unknown="ignore"
                                    ),
                                    VarianceThreshold(threshold=0.01),
                                    MinMaxScaler(),
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
                    X_test = pd.concat(
                        [
                            pd.DataFrame(
                                ct.transform(
                                    predict_data[chosen_cols].drop(
                                        columns=["OS", "OS_days"]
                                    )
                                ),
                                columns=(
                                    ct.transformers_[0][1][0]
                                    .get_feature_names_out()
                                    .tolist()
                                    + [
                                        f"clinical_{i}"
                                        for i in ct.transformers_[1][1][1]
                                        .get_feature_names_out()
                                        .tolist()
                                    ]
                                ),
                            ),
                            predict_data.loc[
                                :, (np.isin(predict_data.columns, ["OS", "OS_days"]))
                            ],
                        ],
                        axis=1,
                    )

                    data_finalized = pd.concat(
                        [
                            pd.concat(
                                [
                                    data_vars_overall_transformed,
                                    data_overall.loc[
                                        :,
                                        (
                                            np.isin(
                                                data_overall.columns, ["OS", "OS_days"]
                                            )
                                        ),
                                    ],
                                ],
                                axis=1,
                            ),
                            X_test,
                        ],
                        axis=0,
                    )
                    mydataset = MyDataset(data_finalized)
                    # const
                    m_length = 128
                    BATCH_SIZE = 250
                    EPOCH = 30
                    lr = 0.01
                    K = 5
                    SEED = 24
                    modalities = available_modalities
                    for outer_split in range(25):
                        train_val = (
                            train_splits.iloc[outer_split, :]
                            .dropna()
                            .values.astype(int)
                        )

                        train_val_df_X, train_val_df_y = (
                            survival_data.iloc[train_val, :],
                            survival_data["OS"].values[train_val],
                        )
                        train_X, val_X, _, _ = train_test_split(
                            train_val_df_X,
                            train_val_df_y,
                            test_size=0.25,
                            random_state=SEED,
                            stratify=train_val_df_y,
                        )
                        train_sampler, val_sampler = (
                            train_X.index.tolist(),
                            val_X.index.tolist(),
                        )
                        test_sampler = [
                            i
                            for i in range(
                                data_vars_overall_transformed.shape[0],
                                data_finalized.shape[0],
                            )
                        ]

                        dataloaders = get_dataloaders(
                            mydataset,
                            train_sampler,
                            val_sampler,
                            test_sampler,
                            BATCH_SIZE,
                        )
                        survmodel = Model(
                            modalities=[
                                modality_remapping[i] for i in available_modalities
                            ],
                            m_length=m_length,
                            dataloaders=dataloaders,
                            fusion_method="attention",
                            input_modality_dim={
                                f"{modality_remapping[i]}": np.sum(
                                    pd.Series(data_finalized.columns)
                                    .str.rsplit("_")
                                    .apply(lambda x: x[0])
                                    .values
                                    == f"{i}"
                                )
                                for i in available_modalities
                            },
                            trade_off=0.3,
                            mode="total",  # only_cox
                            device="cpu",
                        )
                        # Generate run tag
                        run_tag = compose_run_tag(
                            model=survmodel,
                            lr=lr,
                            dataloaders=dataloaders,
                            log_dir="./",
                            suffix="",
                        )
                        fit_args = {
                            "num_epochs": EPOCH,
                            "lr": lr,
                            "info_freq": 2,
                            "log_dir": os.path.join("./", run_tag),
                            "lr_factor": 0.5,
                            "scheduler_patience": 7,
                        }
                        # model fitting
                        survmodel.fit(**fit_args)
                        survmodel.test()
                        train_event = []
                        train_time = []
                        train_hazard = []

                        test_event = []
                        test_time = []
                        test_hazard = []

                        for data, data_label in dataloaders["train"]:
                            out, event, time = survmodel.predict(data, data_label)
                            hazard, representation = out
                            train_event += event.detach().numpy().tolist()
                            train_time += time.detach().numpy().tolist()
                            train_hazard += hazard["hazard"].detach().numpy().tolist()

                        for data, data_label in dataloaders["val"]:
                            out, event, time = survmodel.predict(data, data_label)
                            hazard, representation = out
                            train_event += event.detach().numpy().tolist()
                            train_time += time.detach().numpy().tolist()
                            train_hazard += hazard["hazard"].detach().numpy().tolist()

                        for data, data_label in dataloaders["test"]:
                            out, event, time = survmodel.predict(data, data_label)
                            hazard, representation = out
                            test_event += event.detach().numpy().tolist()
                            test_time += time.detach().numpy().tolist()
                            test_hazard += hazard["hazard"].detach().numpy().tolist()
                        be = BreslowEstimator()
                        be.fit(
                            np.array(train_hazard),
                            np.array(train_event),
                            np.array(train_time),
                        )
                        survival_functions = be.get_survival_function(test_hazard)
                        survival_probabilities = np.stack(
                            [
                                i(np.unique(np.abs(train_time)))
                                for i in survival_functions
                            ]
                        )

                        sf_df = pd.DataFrame(
                            survival_probabilities,
                            columns=np.unique(np.abs(train_time)),
                        )
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
    main()
