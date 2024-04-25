#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import json
import os

import numpy as np
import pandas as pd
from pycox.evaluation.eval_surv import EvalSurv
from scipy.stats import chi2  # type: ignore
from sksurv.nonparametric import kaplan_meier_estimator
from survival_evaluation.utility import to_array


def d_calibration(
    event_indicators,
    predictions,
    bins: int = 5,
) -> dict:
    event_indicators = to_array(event_indicators, to_boolean=True)
    predictions = to_array(predictions)

    # include minimum to catch if probability = 1.
    bin_index = np.minimum(np.floor(predictions * bins), bins - 1).astype(int)
    censored_bin_indexes = bin_index[~event_indicators]
    uncensored_bin_indexes = bin_index[event_indicators]

    censored_predictions = predictions[~event_indicators]
    censored_contribution = 1 - (censored_bin_indexes / bins) * (
        1 / censored_predictions
    )
    censored_following_contribution = 1 / (bins * censored_predictions)

    contribution_pattern = np.tril(np.ones([bins, bins]), k=-1).astype(bool)

    following_contributions = np.matmul(
        censored_following_contribution, contribution_pattern[censored_bin_indexes]
    )
    single_contributions = np.matmul(
        censored_contribution, np.eye(bins)[censored_bin_indexes]
    )
    uncensored_contributions = np.sum(np.eye(bins)[uncensored_bin_indexes], axis=0)
    bin_count = (
        single_contributions + following_contributions + uncensored_contributions
    )
    chi2_statistic = np.sum(
        np.square(bin_count - len(predictions) / bins) / (len(predictions) / bins)
    )
    return dict(
        p_value=1 - chi2.cdf(chi2_statistic, bins - 1),
        test_statistic=chi2_statistic,
        bin_proportions=bin_count / len(predictions),
        censored_contributions=(single_contributions + following_contributions)
        / len(predictions),
        uncensored_contributions=uncensored_contributions / len(predictions),
    )


class EvalSurvDCalib(EvalSurv):
    def __init__(
        self,
        surv,
        durations,
        events,
        censor_surv=None,
        censor_durations=None,
        steps="post",
    ):
        super().__init__(surv, durations, events, censor_surv, censor_durations, steps)

    def d_calibration_(self, bins=5, p_value=False):
        indices = self.idx_at_times(self.durations)
        d_calib = d_calibration(
            self.events,
            np.array(
                [self.surv.iloc[indices[ix], ix] for ix in range(self.events.shape[0])]
            ),
            bins=bins,
        )
        if p_value:
            return d_calib["p_value"]
        else:
            return d_calib["test_statistic"]


with open(os.path.join("./config/", "config.json"), "r") as f:
    config = json.load(f)
config.get("random_state")

ibs_grid_length = 100


antolini_concordance_overall = []
d_calibration_overall = []
ibs_overall = []
cancer_overall = []
project_overall = []
model_overall = []
modalities_overall = []
split_overall = []


# Unimodal
for model in ["kaplan_meier"]:
    for modalities in ["clinical", "gex", "mut", "meth", "cnv", "rppa", "mirna"]:
        for project in ["METABRIC", "TCGA", "TARGET", "ICGC"]:
            for cancer in config[f"{project.lower()}_cancers"]:
                if modalities not in config[f"{project.lower()}_modalities"][cancer]:
                    continue
                cancer_master = pd.read_csv(
                    f"./data_reproduced/{project}/{cancer}_master.csv"
                )
                cancer_train_splits = pd.read_csv(
                    os.path.join(
                        f"./data_reproduced/splits/{project}/{cancer}_train_splits.csv"
                    )
                )
                cancer_test_splits = pd.read_csv(
                    os.path.join(
                        f"./data_reproduced/splits/{project}/{cancer}_test_splits.csv"
                    )
                )

                status = cancer_master["OS"].values
                time = cancer_master["OS_days"].values
                for i in range(cancer_test_splits.shape[0]):
                    test_ix = cancer_test_splits.iloc[i, :].dropna().values.astype(int)
                    train_ix = (
                        cancer_train_splits.iloc[i, :].dropna().values.astype(int)
                    )
                    predictions = pd.read_csv(
                        f"./results_reproduced/survival_functions/{modalities}/{project}/{cancer}/{model}/split_{i + int(model in ['elastic_net', 'rsf', 'kaplan_meier'])}.csv",
                    ).T

                    if model in ["elastic_net", "rsf"] and project != "METABRIC":
                        predictions = predictions.iloc[
                            [
                                i
                                for i in range(len(predictions.index))
                                if "." not in predictions.index[i]
                            ],
                            :,
                        ]
                    elif model in ["elastic_net", "rsf"] and project == "METABRIC":
                        predictions = predictions.iloc[
                            [
                                i
                                for i in range(len(predictions.index))
                                if (predictions.index[i].count(".") < 2)
                                # "." not in predictions.index[i]
                            ],
                            :,
                        ]
                    predictions.index = predictions.index.astype(float)
                    predictions = predictions.sort_index()
                    if np.any(np.isnan(predictions)):
                        # Replace predictions with KM in case there's any missing
                        x, y = kaplan_meier_estimator(
                            status[train_ix].astype(bool), time[train_ix]
                        )
                        predictions = pd.DataFrame(
                            np.stack([y for i in range(test_ix.shape[0])])
                        )
                        predictions.columns = x
                        predictions = predictions.T
                    eval_surv = EvalSurvDCalib(
                        surv=predictions,
                        durations=time[test_ix],
                        events=status[test_ix],
                        censor_surv="km",
                        steps="post",
                    )

                    antolini_concordance_overall.append(eval_surv.concordance_td())

                    d_calibration_overall.append(eval_surv.d_calibration_())

                    ibs_overall.append(
                        eval_surv.integrated_brier_score(
                            time_grid=np.linspace(
                                np.min(time[test_ix]),
                                np.max(time[test_ix]),
                                ibs_grid_length,
                            )
                        )
                    )
                    cancer_overall.append(cancer)
                    project_overall.append(project)
                    model_overall.append(model)
                    modalities_overall.append(modalities)
                    split_overall.append(i)
pd.DataFrame(
    {
        "cancer": cancer_overall,
        "project": project_overall,
        "model": model_overall,
        "modalities": modalities_overall,
        "split": split_overall,
        "antolini_concordance": antolini_concordance_overall,
        "integrated_brier_score": ibs_overall,
        "d_calibration": d_calibration_overall,
    }
).to_csv("./metrics_reproduced/metrics_survboard_finalized_kaplan_meier.csv")

antolini_concordance_overall = []
d_calibration_overall = []
ibs_overall = []
cancer_overall = []
project_overall = []
model_overall = []
modalities_overall = []
split_overall = []


# Unimodal
for model in ["elastic_net", "eh_early", "cox_early", "rsf"]:
    for modalities in ["clinical", "gex", "mut", "rppa", "meth", "cnv", "mirna"]:
        for project in ["METABRIC", "TCGA", "TARGET", "ICGC"]:
            for cancer in config[f"{project.lower()}_cancers"]:
                if modalities not in config[f"{project.lower()}_modalities"][cancer]:
                    continue
                cancer_master = pd.read_csv(
                    f"./data_reproduced/{project}/{cancer}_master.csv"
                )
                cancer_train_splits = pd.read_csv(
                    os.path.join(
                        f"./data_reproduced/splits/{project}/{cancer}_train_splits.csv"
                    )
                )
                cancer_test_splits = pd.read_csv(
                    os.path.join(
                        f"./data_reproduced/splits/{project}/{cancer}_test_splits.csv"
                    )
                )

                status = cancer_master["OS"].values
                time = cancer_master["OS_days"].values
                for i in range(cancer_test_splits.shape[0]):
                    test_ix = cancer_test_splits.iloc[i, :].dropna().values.astype(int)
                    train_ix = (
                        cancer_train_splits.iloc[i, :].dropna().values.astype(int)
                    )
                    predictions = pd.read_csv(
                        f"./results_reproduced/survival_functions/{modalities}/{project}/{cancer}/{model}/split_{i + int(model in ['elastic_net', 'rsf'])}.csv",
                    ).T

                    if model in ["elastic_net", "rsf"] and project != "METABRIC":
                        predictions = predictions.iloc[
                            [
                                i
                                for i in range(len(predictions.index))
                                if "." not in predictions.index[i]
                            ],
                            :,
                        ]
                    elif model in ["elastic_net", "rsf"] and project == "METABRIC":
                        predictions = predictions.iloc[
                            [
                                i
                                for i in range(len(predictions.index))
                                if (predictions.index[i].count(".") < 2)
                                # "." not in predictions.index[i]
                            ],
                            :,
                        ]
                    predictions.index = predictions.index.astype(float)
                    predictions = predictions.sort_index()
                    if np.any(np.isnan(predictions)):
                        # Replace predictions with KM in case there's any missing
                        x, y = kaplan_meier_estimator(
                            status[train_ix].astype(bool), time[train_ix]
                        )
                        predictions = pd.DataFrame(
                            np.stack([y for i in range(test_ix.shape[0])])
                        )
                        predictions.columns = x
                        predictions = predictions.T
                    eval_surv = EvalSurvDCalib(
                        surv=predictions,
                        durations=time[test_ix],
                        events=status[test_ix],
                        censor_surv="km",
                        steps="post",
                    )

                    antolini_concordance_overall.append(eval_surv.concordance_td())

                    d_calibration_overall.append(eval_surv.d_calibration_())

                    ibs_overall.append(
                        eval_surv.integrated_brier_score(
                            time_grid=np.linspace(
                                np.min(time[test_ix]),
                                np.max(time[test_ix]),
                                ibs_grid_length,
                            )
                        )
                    )
                    cancer_overall.append(cancer)
                    project_overall.append(project)
                    model_overall.append(model)
                    modalities_overall.append(modalities)
                    split_overall.append(i)
pd.DataFrame(
    {
        "cancer": cancer_overall,
        "project": project_overall,
        "model": model_overall,
        "modalities": modalities_overall,
        "split": split_overall,
        "antolini_concordance": antolini_concordance_overall,
        "integrated_brier_score": ibs_overall,
        "d_calibration": d_calibration_overall,
    }
).to_csv("./metrics_reproduced/metrics_survboard_finalized_unimodal.csv")

antolini_concordance_overall = []
d_calibration_overall = []
ibs_overall = []
cancer_overall = []
project_overall = []
model_overall = []
modalities_overall = []
split_overall = []

for model in [
    "blockforest",
    "priority_elastic_net",
    "cox_intermediate_concat",
    "cox_late_mean",
    "eh_intermediate_concat",
    "eh_late_mean",
]:
    for modalities in ["clinical_gex", "full"]:
        for project in ["METABRIC", "TCGA", "TARGET", "ICGC"]:
            for cancer in config[f"{project.lower()}_cancers"]:
                cancer_master = pd.read_csv(
                    f"./data_reproduced/{project}/{cancer}_master.csv"
                )
                cancer_train_splits = pd.read_csv(
                    os.path.join(
                        f"./data_reproduced/splits/{project}/{cancer}_train_splits.csv"
                    )
                )
                cancer_test_splits = pd.read_csv(
                    os.path.join(
                        f"./data_reproduced/splits/{project}/{cancer}_test_splits.csv"
                    )
                )

                status = cancer_master["OS"].values
                time = cancer_master["OS_days"].values

                antolini_concordance_cancer = np.zeros(cancer_test_splits.shape[0])
                d_calibration_cancer = np.zeros(cancer_test_splits.shape[0])
                ibs_cancer = np.zeros(cancer_test_splits.shape[0])
                for i in range(cancer_test_splits.shape[0]):
                    test_ix = cancer_test_splits.iloc[i, :].dropna().values.astype(int)
                    train_ix = (
                        cancer_train_splits.iloc[i, :].dropna().values.astype(int)
                    )
                    predictions = pd.read_csv(
                        f"./results_reproduced/survival_functions/{modalities}/{project}/{cancer}/{model}/split_{i + int(model in ['elastic_net', 'rsf', 'priority_elastic_net', 'blockforest'])}.csv",
                    ).T

                    if (
                        model
                        in ["elastic_net", "rsf", "priority_elastic_net", "blockforest"]
                        and project != "METABRIC"
                    ):
                        predictions = predictions.iloc[
                            [
                                i
                                for i in range(len(predictions.index))
                                if "." not in predictions.index[i]
                            ],
                            :,
                        ]
                    elif (
                        model
                        in ["elastic_net", "rsf", "priority_elastic_net", "blockforest"]
                        and project == "METABRIC"
                    ):
                        predictions = predictions.iloc[
                            [
                                i
                                for i in range(len(predictions.index))
                                if (predictions.index[i].count(".") < 2)
                                # "." not in predictions.index[i]
                            ],
                            :,
                        ]
                    predictions.index = predictions.index.astype(float)
                    predictions = predictions.sort_index()
                    if np.any(np.isnan(predictions)):
                        # Replace predictions with KM in case there's any missing
                        x, y = kaplan_meier_estimator(
                            status[train_ix].astype(bool), time[train_ix]
                        )
                        predictions = pd.DataFrame(
                            np.stack([y for i in range(test_ix.shape[0])])
                        )
                        predictions.columns = x
                        predictions = predictions.T
                    eval_surv = EvalSurvDCalib(
                        surv=predictions,
                        durations=time[test_ix],
                        events=status[test_ix],
                        censor_surv="km",
                        steps="post",
                    )

                    antolini_concordance_overall.append(eval_surv.concordance_td())

                    d_calibration_overall.append(eval_surv.d_calibration_())

                    ibs_overall.append(
                        eval_surv.integrated_brier_score(
                            time_grid=np.linspace(
                                np.min(time[test_ix]),
                                np.max(time[test_ix]),
                                ibs_grid_length,
                            )
                        )
                    )
                    cancer_overall.append(cancer)
                    project_overall.append(project)
                    model_overall.append(model)
                    modalities_overall.append(modalities)
                    split_overall.append(i)

pd.DataFrame(
    {
        "cancer": cancer_overall,
        "project": project_overall,
        "model": model_overall,
        "modalities": modalities_overall,
        "split": split_overall,
        "antolini_concordance": antolini_concordance_overall,
        "integrated_brier_score": ibs_overall,
        "d_calibration": d_calibration_overall,
    }
).to_csv("./metrics_reproduced/metrics_survboard_finalized_multimodal.csv")


antolini_concordance_overall = []
d_calibration_overall = []
ibs_overall = []
cancer_overall = []
project_overall = []
model_overall = []
modalities_overall = []
split_overall = []

for model in [
    "priority_elastic_net",
    "cox_intermediate_concat",
    "cox_late_mean",
    "eh_intermediate_concat",
    "eh_late_mean",
]:
    for modalities in ["full_missing"]:
        for project in ["METABRIC", "TCGA", "TARGET", "ICGC"]:
            for cancer in config[f"{project.lower()}_cancers"]:
                cancer_master = pd.read_csv(
                    f"./data_reproduced/{project}/{cancer}_master.csv"
                )
                cancer_train_splits = pd.read_csv(
                    os.path.join(
                        f"./data_reproduced/splits/{project}/{cancer}_train_splits.csv"
                    )
                )
                cancer_test_splits = pd.read_csv(
                    os.path.join(
                        f"./data_reproduced/splits/{project}/{cancer}_test_splits.csv"
                    )
                )

                status = cancer_master["OS"].values
                time = cancer_master["OS_days"].values

                antolini_concordance_cancer = np.zeros(cancer_test_splits.shape[0])
                d_calibration_cancer = np.zeros(cancer_test_splits.shape[0])
                ibs_cancer = np.zeros(cancer_test_splits.shape[0])
                for i in range(cancer_test_splits.shape[0]):
                    test_ix = cancer_test_splits.iloc[i, :].dropna().values.astype(int)
                    train_ix = (
                        cancer_train_splits.iloc[i, :].dropna().values.astype(int)
                    )
                    predictions = pd.read_csv(
                        f"./results_reproduced/survival_functions/{modalities}/{project}/{cancer}/{model}/split_{i + int(model in ['elastic_net', 'rsf', 'priority_elastic_net', 'blockforest'])}.csv",
                    ).T

                    if (
                        model
                        in ["elastic_net", "rsf", "priority_elastic_net", "blockforest"]
                        and project != "METABRIC"
                    ):
                        predictions = predictions.iloc[
                            [
                                i
                                for i in range(len(predictions.index))
                                if "." not in predictions.index[i]
                            ],
                            :,
                        ]
                    elif (
                        model
                        in ["elastic_net", "rsf", "priority_elastic_net", "blockforest"]
                        and project == "METABRIC"
                    ):
                        predictions = predictions.iloc[
                            [
                                i
                                for i in range(len(predictions.index))
                                if (predictions.index[i].count(".") < 2)
                                # "." not in predictions.index[i]
                            ],
                            :,
                        ]
                    predictions.index = predictions.index.astype(float)
                    predictions = predictions.sort_index()
                    if np.any(np.isnan(predictions)):
                        # Replace predictions with KM in case there's any missing
                        x, y = kaplan_meier_estimator(
                            status[train_ix].astype(bool), time[train_ix]
                        )
                        predictions = pd.DataFrame(
                            np.stack([y for i in range(test_ix.shape[0])])
                        )
                        predictions.columns = x
                        predictions = predictions.T
                    eval_surv = EvalSurvDCalib(
                        surv=predictions,
                        durations=time[test_ix],
                        events=status[test_ix],
                        censor_surv="km",
                        steps="post",
                    )

                    antolini_concordance_overall.append(eval_surv.concordance_td())

                    d_calibration_overall.append(eval_surv.d_calibration_())

                    ibs_overall.append(
                        eval_surv.integrated_brier_score(
                            time_grid=np.linspace(
                                np.min(time[test_ix]),
                                np.max(time[test_ix]),
                                ibs_grid_length,
                            )
                        )
                    )
                    cancer_overall.append(cancer)
                    project_overall.append(project)
                    model_overall.append(model)
                    modalities_overall.append(modalities)
                    split_overall.append(i)

pd.DataFrame(
    {
        "cancer": cancer_overall,
        "project": project_overall,
        "model": model_overall,
        "modalities": modalities_overall,
        "split": split_overall,
        "antolini_concordance": antolini_concordance_overall,
        "integrated_brier_score": ibs_overall,
        "d_calibration": d_calibration_overall,
    }
).to_csv("./metrics_reproduced/metrics_survboard_finalized_multimodal_missing.csv")

antolini_concordance_overall = []
d_calibration_overall = []
ibs_overall = []
cancer_overall = []
project_overall = []
model_overall = []
modalities_overall = []
split_overall = []

for model in [
    "blockforest",
    "priority_elastic_net",
    "cox_intermediate_concat",
    "cox_late_mean",
    "eh_intermediate_concat",
    "eh_late_mean",
]:
    for modalities in ["clinical_gex_pancan"]:
        for project in ["TCGA"]:
            for i in range(25):
                predictions_master = (
                    pd.read_csv(
                        f"./results_reproduced/survival_functions/{modalities}/{project}/{model}/split_{i + int(model in ['elastic_net', 'rsf', 'blockforest', 'priority_elastic_net'])}.csv",
                    )
                    # .iloc[offset : (offset + test_ix.shape[0]),]
                    # .T
                )
                offset = 0

                for cancer in config[f"{project.lower()}_cancers"]:
                    print()
                    print(model)
                    print(modalities)
                    print(project)
                    print(cancer)
                    print()
                    cancer_master = pd.read_csv(
                        f"./data_reproduced/{project}/{cancer}_master.csv"
                    )
                    cancer_train_splits = pd.read_csv(
                        os.path.join(
                            f"./data_reproduced/splits/{project}/{cancer}_train_splits.csv"
                        )
                    )
                    cancer_test_splits = pd.read_csv(
                        os.path.join(
                            f"./data_reproduced/splits/{project}/{cancer}_test_splits.csv"
                        )
                    )

                    status = cancer_master["OS"].values
                    time = cancer_master["OS_days"].values

                    antolini_concordance_cancer = np.zeros(cancer_test_splits.shape[0])
                    d_calibration_cancer = np.zeros(cancer_test_splits.shape[0])
                    ibs_cancer = np.zeros(cancer_test_splits.shape[0])
                    cancer_train_splits = pd.read_csv(
                        os.path.join(
                            f"./data_reproduced/splits/{project}/{cancer}_train_splits.csv"
                        )
                    )
                    cancer_test_splits = pd.read_csv(
                        os.path.join(
                            f"./data_reproduced/splits/{project}/{cancer}_test_splits.csv"
                        )
                    )
                    test_ix = cancer_test_splits.iloc[i, :].dropna().values.astype(int)
                    train_ix = (
                        cancer_train_splits.iloc[i, :].dropna().values.astype(int)
                    )
                    predictions = predictions_master.iloc[
                        offset : (offset + test_ix.shape[0]),
                    ].T
                    if model in [
                        "elastic_net",
                        "rsf",
                        "priority_elastic_net",
                        "blockforest",
                    ]:
                        predictions = predictions.iloc[
                            [
                                i
                                for i in range(len(predictions.index))
                                if "." not in predictions.index[i]
                            ],
                            :,
                        ]
                    offset += test_ix.shape[0]

                    predictions.index = predictions.index.astype(int)
                    predictions = predictions.sort_index()
                    if np.any(np.isnan(predictions)):
                        # Replace predictions with KM in case there's any missing
                        x, y = kaplan_meier_estimator(
                            status[train_ix].astype(bool), time[train_ix]
                        )
                        predictions = pd.DataFrame(
                            np.stack([y for i in range(test_ix.shape[0])])
                        )
                        predictions.columns = x
                        predictions = predictions.T
                    eval_surv = EvalSurvDCalib(
                        surv=predictions,
                        durations=time[test_ix],
                        events=status[test_ix],
                        censor_surv="km",
                        steps="post",
                    )

                    antolini_concordance_overall.append(eval_surv.concordance_td())

                    d_calibration_overall.append(eval_surv.d_calibration_())

                    ibs_overall.append(
                        eval_surv.integrated_brier_score(
                            time_grid=np.linspace(
                                np.min(time[test_ix]),
                                np.max(time[test_ix]),
                                ibs_grid_length,
                            )
                        )
                    )
                    cancer_overall.append(cancer)
                    project_overall.append(project)
                    model_overall.append(model)
                    modalities_overall.append(modalities)
                    split_overall.append(i)

pd.DataFrame(
    {
        "cancer": cancer_overall,
        "project": project_overall,
        "model": model_overall,
        "modalities": modalities_overall,
        "split": split_overall,
        "antolini_concordance": antolini_concordance_overall,
        "integrated_brier_score": ibs_overall,
        "d_calibration": d_calibration_overall,
    }
).to_csv("./metrics_reproduced/metrics_survboard_finalized_pancancer.csv")
