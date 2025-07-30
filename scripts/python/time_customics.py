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
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.utils import parallel_backend
from sksurv.linear_model.coxph import BreslowEstimator
from survboard.CustOmics.src.network.customics import CustOMICS
from survboard.CustOmics.src.tools.prepare_dataset import prepare_dataset
from survboard.CustOmics.src.tools.utils import get_sub_omics_df
from survboard.python.model.multimodal_survival_pred import setup_seed
from survboard.python.utils.misc_utils import seed_torch  # get_blocks_gdp,

with open(snakemake.log[0], "w") as f:
    sys.stderr = f
    sys.stdout = f
    with open(os.path.join("./config/", "config.json"), "r") as f:
        config = json.load(f)

    g = np.random.default_rng(config.get("random_state"))
    seed_torch(config.get("random_state"))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    setup_seed(config.get("random_state"))
    for fusion in ["customics"]:
        for model_type in ["customics"]:
            for project in ["TCGA"]:
                for cancer in [snakemake.wildcards["cancer"]]:
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

                    data_overall_vars = data.iloc[:, np.where(column_types != "OS")[0]]

                    ct = ColumnTransformer(
                        [
                            (
                                "numerical",
                                make_pipeline(
                                    VarianceThreshold(threshold=0.0), MinMaxScaler()
                                ),
                                np.where(data_overall_vars.dtypes != "object")[0],
                            ),
                            (
                                "categorical",
                                make_pipeline(
                                    OneHotEncoder(
                                        sparse=False, handle_unknown="ignore"
                                    ),
                                    VarianceThreshold(threshold=0.0),
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
                    data_finalized = pd.concat(
                        [
                            data_vars_overall_transformed,
                            data.loc[:, (np.isin(data.columns, ["OS", "OS_days"]))],
                        ],
                        axis=1,
                    )
                    data_finalized["label_dummy"] = 1
                    modalities = available_modalities
                    feature_names = data_finalized.columns
                    column_types = (
                        pd.Series(feature_names)
                        .str.rsplit("_")
                        .apply(lambda x: x[0])
                        .values
                    )

                    available_modalities = [
                        i for i in np.unique(column_types) if i not in ["OS", "label"]
                    ]

                    omics_df = {
                        f"{i}": data_finalized[
                            [q for q in data_finalized.columns if f"{i}_" in q]
                        ]
                        for i in available_modalities
                    }
                    clinical_df = data_finalized[["OS_days", "OS", "label_dummy"]]

                    samples_train = [i for i in range(omics_df["gex"].shape[0])]

                    samples_train, samples_val = train_test_split(
                        samples_train,
                        test_size=0.2,
                        random_state=config.get("seed"),
                        stratify=clinical_df[["OS"]].values[np.array(samples_train)],
                    )

                    omics_train = get_sub_omics_df(omics_df, samples_train)
                    omics_val = get_sub_omics_df(omics_df, samples_val)

                    x_dim = [
                        omics_df[omic_source].shape[1]
                        for omic_source in omics_df.keys()
                    ]
                    # print(omics_df["gex"])
                    # raise ValueError
                    # print(x_dim)
                    # print(data_finalized.head())
                    # print(data_finalized.shape)
                    # print(np.sum(np.isnan(data_finalized)))
                    # raise ValueError

                    batch_size = 32
                    n_epochs = 20
                    device = torch.device("cpu")
                    label = "label_dummy"
                    event = "OS"
                    surv_time = "OS_days"

                    task = "survival"
                    sources = available_modalities

                    hidden_dim = [1024, 512, 256]
                    central_dim = [2048, 1024, 512, 256]
                    rep_dim = 128
                    latent_dim = 128
                    num_classes = 1
                    dropout = 0.2
                    beta = 1
                    lambda_classif = 0
                    classifier_dim = [256, 128]
                    lambda_survival = 5
                    survival_dim = [64, 32]
                    source_params = {}
                    central_params = {
                        "hidden_dim": central_dim,
                        "latent_dim": latent_dim,
                        "norm": True,
                        "dropout": dropout,
                        "beta": beta,
                    }
                    classif_params = {
                        "n_class": num_classes,
                        "lambda": lambda_classif,
                        "hidden_layers": classifier_dim,
                        "dropout": dropout,
                    }
                    surv_params = {
                        "lambda": lambda_survival,
                        "dims": survival_dim,
                        "activation": "SELU",
                        "l2_reg": 1e-2,
                        "norm": True,
                        "dropout": dropout,
                    }
                    for i, source in enumerate(sources):
                        source_params[source] = {
                            "input_dim": x_dim[i],
                            "hidden_dim": hidden_dim,
                            "latent_dim": rep_dim,
                            "norm": True,
                            "dropout": 0.2,
                        }
                    train_params = {"switch": 10, "lr": 1e-3}

                    model = CustOMICS(
                        source_params=source_params,
                        central_params=central_params,
                        classif_params=classif_params,
                        surv_params=surv_params,
                        train_params=train_params,
                        device=device,
                    ).to(device)
                    model.fit(
                        omics_train=omics_train,
                        clinical_df=clinical_df,
                        label=label,
                        event=event,
                        surv_time=surv_time,
                        omics_val=omics_val,
                        batch_size=batch_size,
                        n_epochs=n_epochs,
                        verbose=True,
                    )

                    train_risk = (
                        model.predict_risk(omics_train).detach().numpy().ravel()
                    )
                    val_risk = model.predict_risk(omics_val).detach().numpy().ravel()

                    train_time = clinical_df[["OS_days"]].values[samples_train].ravel()
                    val_time = clinical_df[["OS_days"]].values[samples_val].ravel()

                    train_event = clinical_df[["OS"]].values[samples_train].ravel()
                    val_event = clinical_df[["OS"]].values[samples_val].ravel()

                    be = BreslowEstimator()
                    be.fit(
                        np.concatenate([train_risk, val_risk]),
                        np.concatenate([train_event, val_event]),
                        np.concatenate([train_time, val_time]),
                    )
                    pathlib.Path(
                        f"results_reproduced/timings/customics_{snakemake.wildcards['cancer']}"
                    ).touch()

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
