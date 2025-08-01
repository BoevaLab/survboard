#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import platform
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
from pycox.evaluation.eval_surv import EvalSurv
from scipy.stats import chi2  # type: ignore
from sksurv.nonparametric import kaplan_meier_estimator
from survival_evaluation.utility import to_array


def _group_loop(n, surv_idx, durations, events, di, ni):
    idx = 0
    for i in range(n):
        idx += durations[i] != surv_idx[idx]
        di[idx] += events[i]
        ni[idx] += 1
    return di, ni


def kaplan_meier(durations, events, start_duration=0):
    """A very simple Kaplan-Meier fitter. For a more complete implementation
    see `lifelines`.

    Arguments:
        durations {np.array} -- durations array
        events {np.arrray} -- events array 0/1

    Keyword Arguments:
        start_duration {int} -- Time start as `start_duration`. (default: {0})

    Returns:
        pd.Series -- Kaplan-Meier estimates.
    """
    n = len(durations)
    assert n == len(events)
    if start_duration > durations.min():
        warnings.warn(
            f"start_duration {start_duration} is larger than minimum duration {durations.min()}. "
            "If intentional, consider changing start_duration when calling kaplan_meier."
        )
    order = np.argsort(durations)
    durations = durations[order]
    events = events[order]
    surv_idx = np.unique(durations)
    ni = np.zeros(len(surv_idx), dtype="int")
    di = np.zeros_like(ni)
    di, ni = _group_loop(n, surv_idx, durations, events, di, ni)
    ni = n - ni.cumsum()
    ni[1:] = ni[:-1]
    ni[0] = n
    survive = 1 - di / ni
    zero_survive = survive == 0
    if zero_survive.any():
        i = np.argmax(zero_survive)
        surv = np.zeros_like(survive)
        surv[:i] = np.exp(np.log(survive[:i]).cumsum())
        # surv[i:] = surv[i-1]
        surv[i:] = 0.0
    else:
        surv = np.exp(np.log(1 - di / ni).cumsum())
    if start_duration < surv_idx.min():
        tmp = np.ones(len(surv) + 1, dtype=surv.dtype)
        tmp[1:] = surv
        surv = tmp
        tmp = np.zeros(len(surv_idx) + 1, dtype=surv_idx.dtype)
        tmp[1:] = surv_idx
        surv_idx = tmp
    surv = pd.Series(surv, surv_idx)
    return surv


def make_cuts(n_cuts, scheme, durations, events, min_=0.0, dtype="float64"):
    if scheme == "equidistant":
        cuts = cuts_equidistant(durations.max(), n_cuts, min_, dtype)
    elif scheme == "quantiles":
        cuts = cuts_quantiles(durations, events, n_cuts, min_, dtype)
    else:
        raise ValueError(f"Got invalid `scheme` {scheme}.")
    if (np.diff(cuts) == 0).any():
        raise ValueError("cuts are not unique.")
    return cuts


def _values_if_series(x):
    if type(x) is pd.Series:
        return x.values
    return x


def cuts_equidistant(max_, num, min_=0.0, dtype="float64"):
    return np.linspace(min_, max_, num, dtype=dtype)


def cuts_quantiles(durations, events, num, min_=0.0, dtype="float64"):
    # raise NotImplementedError
    """
    If min_ = None, we will use durations.min() for the first cut.
    """
    km = kaplan_meier(durations, events)
    surv_est, surv_durations = km.values, km.index.values
    s_cuts = np.linspace(km.values.min(), km.values.max(), num)
    cuts_idx = np.searchsorted(surv_est[::-1], s_cuts)[::-1]
    cuts = surv_durations[::-1][cuts_idx]
    cuts = np.unique(cuts)
    if len(cuts) != num:
        warnings.warn(
            f"cuts are not unique, continue with {len(cuts)} cuts instead of {num}"
        )
    cuts[0] = durations.min() if min_ is None else min_
    assert cuts[-1] == durations.max(), "something wrong..."
    return cuts.astype(dtype)


def _is_monotonic_increasing(x):
    assert len(x.shape) == 1, "Only works for 1d"
    return (x[1:] >= x[:-1]).all()


def bin_numerical(x, right_cuts, error_on_larger=False):
    """
    Discretize x into bins defined by right_cuts (needs to be sorted).
    If right_cuts = [1, 2], we have bins (-inf, 1], (1, 2], (2, inf).
    error_on_larger results in a ValueError if x contains larger
    values than right_cuts.

    Returns index of bins.
    To optaine values do righ_cuts[bin_numerica(x, right_cuts)].
    """
    assert _is_monotonic_increasing(right_cuts), "Need `right_cuts` to be sorted."
    bins = np.searchsorted(right_cuts, x, side="left")
    if bins.max() == right_cuts.size:
        if error_on_larger:
            raise ValueError("x contrains larger values than right_cuts.")
    return bins


def discretize(x, cuts, side="right", error_on_larger=False):
    """Discretize x to cuts.

    Arguments:
        x {np.array} -- Array of times.
        cuts {np.array} -- Sortet array of discrete times.

    Keyword Arguments:
        side {str} -- If we shold round down or up (left, right) (default: {'right'})
        error_on_larger {bool} -- If we shold return an error if we pass higher values
            than cuts (default: {False})

    Returns:
        np.array -- Discretized values.
    """
    if side not in ["right", "left"]:
        raise ValueError("side argument needs to be right or left.")
    bins = bin_numerical(x, cuts, error_on_larger)
    if side == "right":
        cuts = np.concatenate((cuts, np.array([np.inf])))
        return cuts[bins]
    bins_cut = bins.copy()
    bins_cut[bins_cut == cuts.size] = -1
    exact = cuts[bins_cut] == x
    left_bins = bins - 1 + exact
    vals = cuts[left_bins]
    vals[left_bins == -1] = -np.inf
    return vals


class _OnlyTransform:
    """Abstract class for sklearn preprocessing methods.
    Only implements fit and fit_transform.
    """

    def fit(self, *args):
        return self

    def transform(self, *args):
        raise NotImplementedError

    def fit_transform(self, *args):
        return self.fit(*args).transform(*args)


class DiscretizeUnknownC(_OnlyTransform):
    """Implementation of scheme 2.

    cuts should be [t0, t1, ..., t_m], where t_m is right sensored value.
    """

    def __init__(self, cuts, right_censor=False, censor_side="left"):
        self.cuts = cuts
        self.right_censor = right_censor
        self.censor_side = censor_side

    def transform(self, duration, event):
        dtype_event = event.dtype
        event = event.astype("bool")
        if self.right_censor:
            duration = duration.copy()
            censor = duration > self.cuts.max()
            duration[censor] = self.cuts.max()
            event[censor] = False
        if duration.max() > self.cuts.max():
            raise ValueError(
                "`duration` contains larger values than cuts. Set `right_censor`=True to censor these"
            )
        td = np.zeros_like(duration)
        c = event == False
        td[event] = discretize(
            duration[event], self.cuts, side="right", error_on_larger=True
        )
        if c.any():
            td[c] = discretize(
                duration[c], self.cuts, side=self.censor_side, error_on_larger=True
            )
        return td, event.astype(dtype_event)


def duration_idx_map(duration):
    duration = np.unique(duration)
    duration = np.sort(duration)
    idx = np.arange(duration.shape[0])
    return {d: i for i, d in zip(idx, duration)}


class Duration2Idx(_OnlyTransform):
    def __init__(self, durations=None):
        self.durations = durations
        if durations is None:
            raise NotImplementedError()
        if self.durations is not None:
            self.duration_to_idx = self._make_map(self.durations)

    @staticmethod
    def _make_map(durations):
        return np.vectorize(duration_idx_map(durations).get)

    def transform(self, duration, y=None):
        if duration.dtype is not self.durations.dtype:
            raise ValueError("Need `time` to have same type as `self.durations`.")
        idx = self.duration_to_idx(duration)
        if np.isnan(idx).any():
            raise ValueError("Encountered `nans` in transformed indexes.")
        return idx


class IdxDiscUnknownC:
    """Get indexed for discrete data using cuts.

    Arguments:
        cuts {np.array} -- Array or right cuts.

    Keyword Arguments:
        label_cols {tuple} -- Name of label columns in dataframe (default: {None}).
    """

    def __init__(self, cuts, label_cols=None, censor_side="left"):
        self.cuts = cuts
        self.duc = DiscretizeUnknownC(cuts, right_censor=True, censor_side=censor_side)
        self.di = Duration2Idx(cuts)
        self.label_cols = label_cols

    def transform(self, time, d):
        time, d = self.duc.transform(time, d)
        idx = self.di.transform(time)
        return idx, d

    def transform_df(self, df):
        if self.label_cols is None:
            raise RuntimeError(
                "Need to set 'label_cols' to use this. Use 'transform instead'"
            )
        col_duration, col_event = self.label_cols
        time = df[col_duration].values
        d = df[col_event].values
        return self.transform(time, d)


class LabTransDiscreteTime:
    """
    Discretize continuous (duration, event) pairs based on a set of cut points.
    One can either determine the cut points in form of passing an array to this class,
    or one can obtain cut points based on the training data.

    The discretization learned from fitting to data will move censorings to the left cut point,
    and events to right cut point.

    Arguments:
        cuts {int, array} -- Defining cut points, either the number of cuts, or the actual cut points.

    Keyword Arguments:
        scheme {str} -- Scheme used for discretization. Either 'equidistant' or 'quantiles'
            (default: {'equidistant})
        min_ {float} -- Starting duration (default: {0.})
        dtype {str, dtype} -- dtype of discretization.
    """

    def __init__(self, cuts, scheme="equidistant", min_=0.0, dtype=None):
        self._cuts = cuts
        self._scheme = scheme
        self._min = min_
        self._dtype_init = dtype
        self._predefined_cuts = False
        self.cuts = None
        if hasattr(cuts, "__iter__"):
            if type(cuts) is list:
                cuts = np.array(cuts)
            self.cuts = cuts
            self.idu = IdxDiscUnknownC(self.cuts)
            assert dtype is None, "Need `dtype` to be `None` for specified cuts"
            self._dtype = type(self.cuts[0])
            self._dtype_init = self._dtype
            self._predefined_cuts = True

    def fit(self, durations, events):
        if self._predefined_cuts:
            warnings.warn(
                "Calling fit method, when 'cuts' are already defined. Leaving cuts unchanged."
            )
            return self
        self._dtype = self._dtype_init
        if self._dtype is None:
            if isinstance(durations[0], np.floating):
                self._dtype = durations.dtype
            else:
                self._dtype = np.dtype("float64")
        durations = durations.astype(self._dtype)
        self.cuts = make_cuts(
            self._cuts, self._scheme, durations, events, self._min, self._dtype
        )
        self.idu = IdxDiscUnknownC(self.cuts)
        return self

    def fit_transform(self, durations, events):
        self.fit(durations, events)
        idx_durations, events = self.transform(durations, events)
        return idx_durations, events

    def transform(self, durations, events):
        durations = _values_if_series(durations)
        durations = durations.astype(self._dtype)
        events = _values_if_series(events)
        idx_durations, events = self.idu.transform(durations, events)
        return idx_durations.astype("int64"), events.astype("float32")

    @property
    def out_features(self):
        """Returns the number of output features that should be used in the torch model.

        Returns:
            [int] -- Number of output features.
        """
        if self.cuts is None:
            raise ValueError("Need to call `fit` before this is accessible.")
        return len(self.cuts)


def transform_discrete_time(time, event, n_durations, scheme):
    trans = LabTransDiscreteTime(cuts=n_durations, scheme=scheme)
    y = trans.fit_transform(time, event)
    return (y, trans.cuts)


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
                    if model in ["elastic_net", "rsf"]:
                        predictions = pd.read_csv(
                            f"./results_reproduced/survival_functions/{modalities}/{project}/{cancer}/{model}/split_{i + int(model in ['elastic_net', 'rsf'])}.csv",
                        ).T
                    else:
                        predictions = pd.read_csv(
                            f"./results_reproduced/survival_functions_consolidated_csv/{project}/{cancer}/unimodal/split_{i + int(model in ['elastic_net', 'rsf'])}.csv",
                        )
                        predictions = predictions.loc[
                            predictions.modality == modalities, :
                        ]
                        predictions.model_type = predictions.model_type + "_early"
                        predictions = (
                            predictions.loc[predictions.model_type == model]
                            .iloc[:, :-5]
                            .T
                        )

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
failures_overall = []

for model in [
    "blockforest",
    "priority_elastic_net",
    "salmon_salmon",
    "multimodal_nsclc",
    "survival_net_survival_net",
    "gdp_gdp",
    "customics_customics",
    "multimodal_survival_pred_multimodal_survival_pred",
    "cox_intermediate_concat",
    "cox_late_mean",
    "eh_intermediate_concat",
    "eh_late_mean",
]:
    for modalities in ["full"]:
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
                for i in range(0, cancer_test_splits.shape[0]):
                    test_ix = cancer_test_splits.iloc[i, :].dropna().values.astype(int)
                    train_ix = (
                        cancer_train_splits.iloc[i, :].dropna().values.astype(int)
                    )
                    if (
                        model in ["survival_net_survival_net", "gdp_gdp"]
                        and project != "METABRIC"
                    ):
                        predictions = pd.read_csv(
                            f"./results_reproduced/survival_functions/{modalities}/{project}/{cancer}/{model}/split_{i}.csv",
                        ).T
                    elif model in [
                        "cox_late_mean",
                        "cox_intermediate_concat",
                        "eh_late_mean",
                        "eh_intermediate_concat",
                    ]:
                        predictions = pd.read_csv(
                            f"./results_reproduced/survival_functions_consolidated_csv/{project}/{cancer}/{modalities}/split_{i}.csv",
                        )
                        predictions.model_type = (
                            predictions.model_type + "_" + predictions.fusion
                        )
                        predictions = (
                            predictions.loc[predictions.model_type == model]
                            .iloc[:, :-6]
                            .T
                        )
                    else:
                        predictions = pd.read_csv(
                            f"./results_reproduced/survival_functions/{modalities}/{project}/{cancer}/{model}/split_{i + int(model in ['multimodal_nsclc', 'elastic_net', 'rsf', 'priority_elastic_net', 'blockforest'])}.csv",
                        ).T
                    if np.all(
                        predictions.iloc[:, 0] == predictions.iloc[:, 1]
                    ) and np.all(predictions.iloc[:, 1] == predictions.iloc[:, 2]):
                        failures_overall.append(1)
                    else:
                        failures_overall.append(0)

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
                            ],
                            :,
                        ]
                    predictions.index = predictions.index.astype(float)

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
                    predictions = predictions.sort_index()
                    eval_surv = EvalSurvDCalib(
                        surv=predictions,
                        durations=time[test_ix],
                        events=status[test_ix],
                        censor_surv="km",
                        steps="post",
                    )
                    try:
                        antolini_concordance_overall.append(eval_surv.concordance_td())
                    except AssertionError:
                        print(project)
                        print(cancer)
                        print(i)
                        print(model)

                        print(predictions.shape)
                        print(time[test_ix].shape)

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


for model in [
    "blockforest",
    "priority_elastic_net",
    "salmon_salmon",
    "multimodal_nsclc",
    "survival_net_survival_net",
    "gdp_gdp",
    "customics_customics",
    "multimodal_survival_pred_multimodal_survival_pred",
    "cox_intermediate_concat",
    "cox_late_mean",
    "eh_intermediate_concat",
    "eh_late_mean",
]:
    for modalities in ["clinical_gex"]:
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
                    if (
                        model in ["survival_net_survival_net", "gdp", "gdp"]
                        and project != "METABRIC"
                    ):
                        predictions = pd.read_csv(
                            f"./results_reproduced/survival_functions/{modalities}/{project}/{cancer}/{model}/split_{i}.csv",
                        ).T
                    elif model in [
                        "cox_late_mean",
                        "cox_intermediate_concat",
                        "eh_late_mean",
                        "eh_intermediate_concat",
                    ]:
                        predictions = pd.read_csv(
                            f"./results_reproduced/survival_functions_consolidated_csv/{project}/{cancer}/{modalities}/split_{i}.csv",
                        )
                        predictions.model_type = (
                            predictions.model_type + "_" + predictions.fusion
                        )
                        predictions = (
                            predictions.loc[predictions.model_type == model]
                            .iloc[:, :-6]
                            .T
                        )
                    else:
                        predictions = pd.read_csv(
                            f"./results_reproduced/survival_functions/{modalities}/{project}/{cancer}/{model}/split_{i + int(model in ['multimodal_nsclc', 'elastic_net', 'rsf', 'priority_elastic_net', 'blockforest'])}.csv",
                        ).T
                    if np.all(
                        predictions.iloc[:, 0] == predictions.iloc[:, 1]
                    ) and np.all(predictions.iloc[:, 1] == predictions.iloc[:, 2]):
                        failures_overall.append(1)
                    else:
                        failures_overall.append(0)
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
                            ],
                            :,
                        ]
                    predictions.index = predictions.index.astype(float)
                    predictions = predictions.sort_index()
                    if "discrete_time" in model:
                        y, cuts = transform_discrete_time(
                            time[train_ix], status[train_ix], 30, "equidistant"
                        )
                        predictions = predictions.loc[
                            [i in cuts for i in predictions.index], :
                        ]
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
        "failures": failures_overall,
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
    "eh_intermediate_concat",
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

                    if (
                        model
                        in ["elastic_net", "rsf", "priority_elastic_net", "blockforest"]
                        and project != "METABRIC"
                    ):
                        predictions = pd.read_csv(
                            f"./results_reproduced/survival_functions/{modalities}/{project}/{cancer}/{model}/split_{i + int(model in ['elastic_net', 'rsf', 'priority_elastic_net', 'blockforest'])}.csv",
                        ).T
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
                        predictions = pd.read_csv(
                            f"./results_reproduced/survival_functions/{modalities}/{project}/{cancer}/{model}/split_{i + int(model in ['elastic_net', 'rsf', 'priority_elastic_net', 'blockforest'])}.csv",
                        ).T
                        predictions = predictions.iloc[
                            [
                                i
                                for i in range(len(predictions.index))
                                if (predictions.index[i].count(".") < 2)
                            ],
                            :,
                        ]
                    elif model in [
                        "cox_late_mean",
                        "cox_intermediate_concat",
                        "eh_late_mean",
                        "eh_intermediate_concat",
                    ]:
                        predictions = pd.read_csv(
                            f"./results_reproduced/survival_functions_consolidated_csv/{project}/{cancer}/{modalities}/split_{i}.csv",
                        )
                        predictions.model_type = (
                            predictions.model_type + "_" + predictions.fusion
                        )
                        predictions = (
                            predictions.loc[predictions.model_type == model]
                            .iloc[:, :-6]
                            .T
                        )
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
    "eh_intermediate_concat",
]:
    for modalities in ["clinical_gex_pancan"]:
        for project in ["TCGA"]:
            for i in range(25):
                if model in [
                    "cox_late_mean",
                    "cox_intermediate_concat",
                    "eh_late_mean",
                    "eh_intermediate_concat",
                ]:
                    predictions_master = pd.read_csv(
                        f"./results_reproduced/survival_functions_consolidated_csv/{project}/pancancer/pancancer/split_{i}.csv",
                    )
                    predictions_master.model_type = (
                        predictions_master.model_type + "_" + predictions_master.fusion
                    )
                    predictions_master = predictions_master.loc[
                        predictions_master.model_type == model
                    ].iloc[:, :-6]
                else:
                    predictions_master = pd.read_csv(
                        f"./results_reproduced/survival_functions/{modalities}/{project}/{model}/split_{i + int(model in ['elastic_net', 'rsf', 'blockforest', 'priority_elastic_net'])}.csv",
                    )
                offset = 0

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
).to_csv("./metrics_reproduced/metrics_survboard_finalized_pancancer.csv")


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
    "blockforest",
    "salmon_salmon",
    "multimodal_nsclc",
    "survival_net_survival_net",
    "gdp_gdp",
    "customics_customics",
    "multimodal_survival_pred_multimodal_survival_pred",
    "cox_intermediate_concat",
    "cox_late_mean",
    "eh_intermediate_concat",
    "eh_late_mean",
]:
    for modalities in ["transfer"]:
        for project in ["validation"]:
            for cancer in ["LIHC", "PAAD"]:
                if cancer == "LIHC":
                    cancer_train_master = pd.read_csv(
                        f"./data_reproduced/TCGA/LIHC_master.csv"
                    )
                    cancer_test_master = pd.read_csv(
                        f"./data_reproduced/ICGC/LIRI-JP_master.csv"
                    )
                    cancer_train_splits = pd.read_csv(
                        os.path.join(
                            f"./data_reproduced/splits/TCGA/LIHC_train_splits.csv"
                        )
                    )
                else:
                    cancer_train_master = pd.read_csv(
                        f"./data_reproduced/ICGC/PACA-CA_master.csv"
                    )
                    cancer_test_master = pd.read_csv(
                        f"./data_reproduced/TCGA/PAAD_master.csv"
                    )
                    cancer_train_splits = pd.read_csv(
                        os.path.join(
                            f"./data_reproduced/splits/ICGC/PACA-CA_train_splits.csv"
                        )
                    )

                train_status = cancer_train_master["OS"].values
                train_time = cancer_train_master["OS_days"].values

                test_status = cancer_test_master["OS"].values
                test_time = cancer_test_master["OS_days"].values

                antolini_concordance_cancer = np.zeros(25)
                d_calibration_cancer = np.zeros(25)
                ibs_cancer = np.zeros(25)
                for i in range(25):
                    train_ix = (
                        cancer_train_splits.iloc[i, :].dropna().values.astype(int)
                    )

                    predictions = pd.read_csv(
                        f"./results_reproduced/survival_functions/{modalities}/{project}/{cancer}/{model}/split_{i + int(model in ['multimodal_nsclc', 'elastic_net', 'rsf', 'priority_elastic_net', 'blockforest'])}.csv",
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
                            ],
                            :,
                        ]
                    predictions.index = predictions.index.astype(float)
                    predictions = predictions.sort_index()
                    if np.any(np.isnan(predictions)):
                        # Replace predictions with KM in case there's any missing
                        x, y = kaplan_meier_estimator(
                            train_status[train_ix].astype(bool), train_time[train_ix]
                        )
                        predictions = pd.DataFrame(
                            np.stack([y for i in range(test_time.shape[0])])
                        )
                        predictions.columns = x
                        predictions = predictions.T
                    eval_surv = EvalSurvDCalib(
                        surv=predictions,
                        durations=test_time,
                        events=test_status,
                        censor_surv="km",
                        steps="post",
                    )

                    antolini_concordance_overall.append(eval_surv.concordance_td())

                    d_calibration_overall.append(eval_surv.d_calibration_())

                    ibs_overall.append(
                        eval_surv.integrated_brier_score(
                            time_grid=np.linspace(
                                np.min(test_time),
                                np.max(test_time),
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
).to_csv("./metrics_reproduced/metrics_survboard_finalized_transfer.csv")


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
