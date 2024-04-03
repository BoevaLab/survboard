import math
import numbers
import os
import random
import time
from contextlib import contextmanager
from traceback import format_exc

import numpy as np
import pandas as pd
import torch
from joblib import logger
from numba import jit
from scipy.integrate import quadrature
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection._split import _RepeatedSplits
from sklearn.model_selection._validation import _score
from sklearn.utils.metaestimators import _safe_split
from sklearn.utils.validation import _check_fit_params, _num_samples
from skorch.callbacks import EpochScoring
from skorch.dataset import ValidSplit, get_len
from skorch.utils import data_from_dataset, is_skorch_dataset, to_device, to_numpy
from torch.nn.modules.loss import _Loss


def multimodal_dropout(x, p_multimodal_dropout, blocks, upweight=True):
    for block in blocks:
        if not torch.all(x[:, block] == 0):
            msk = torch.where((torch.rand(x.shape[0]) <= p_multimodal_dropout).long())[
                0
            ]
            x[:, torch.tensor(block)][msk, :] = torch.zeros(
                x[:, torch.tensor(block)][msk, :].shape
            )

    if upweight:
        x = x / (1 - p_multimodal_dropout)
    return x


def _fit_and_score_survival(
    estimator,
    X,
    y,
    scorer,
    train,
    test,
    verbose,
    parameters,
    fit_params,
    return_train_score=False,
    return_parameters=False,
    return_n_test_samples=False,
    return_times=False,
    return_estimator=False,
    split_progress=None,
    candidate_progress=None,
    error_score=np.nan,
):
    """Fit estimator and compute scores for a given dataset split.
    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.
    X : array-like of shape (n_samples, n_features)
        The data to fit.
    y : array-like of shape (n_samples,) or (n_samples, n_outputs) or None
        The target variable to try to predict in the case of
        supervised learning.
    scorer : A single callable or dict mapping scorer name to the callable
        If it is a single callable, the return value for ``train_scores`` and
        ``test_scores`` is a single float.
        For a dict, it should be one mapping the scorer name to the scorer
        callable object / function.
        The callable object / fn should have signature
        ``scorer(estimator, X, y)``.
    train : array-like of shape (n_train_samples,)
        Indices of training samples.
    test : array-like of shape (n_test_samples,)
        Indices of test samples.
    verbose : int
        The verbosity level.
    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised.
        If a numeric value is given, FitFailedWarning is raised.
    parameters : dict or None
        Parameters to be set on the estimator.
    fit_params : dict or None
        Parameters that will be passed to ``estimator.fit``.
    return_train_score : bool, default=False
        Compute and return score on training set.
    return_parameters : bool, default=False
        Return parameters that has been used for the estimator.
    split_progress : {list, tuple} of int, default=None
        A list or tuple of format (<current_split_id>, <total_num_of_splits>).
    candidate_progress : {list, tuple} of int, default=None
        A list or tuple of format
        (<current_candidate_id>, <total_number_of_candidates>).
    return_n_test_samples : bool, default=False
        Whether to return the ``n_test_samples``.
    return_times : bool, default=False
        Whether to return the fit/score times.
    return_estimator : bool, default=False
        Whether to return the fitted estimator.
    Returns
    -------
    result : dict with the following attributes
        train_scores : dict of scorer name -> float
            Score on training set (for all the scorers),
            returned only if `return_train_score` is `True`.
        test_scores : dict of scorer name -> float
            Score on testing set (for all the scorers).
        n_test_samples : int
            Number of test samples.
        fit_time : float
            Time spent for fitting in seconds.
        score_time : float
            Time spent for scoring in seconds.
        parameters : dict or None
            The parameters that have been evaluated.
        estimator : estimator object
            The fitted estimator.
        fit_error : str or None
            Traceback str if the fit failed, None if the fit succeeded.
    """
    if not isinstance(error_score, numbers.Number) and error_score != "raise":
        raise ValueError(
            "error_score must be the string 'raise' or a numeric value. "
            "(Hint: if using 'raise', please make sure that it has been "
            "spelled correctly.)"
        )

    progress_msg = ""
    if verbose > 2:
        if split_progress is not None:
            progress_msg = f" {split_progress[0]+1}/{split_progress[1]}"
        if candidate_progress and verbose > 9:
            progress_msg += f"; {candidate_progress[0]+1}/{candidate_progress[1]}"

    if verbose > 1:
        if parameters is None:
            params_msg = ""
        else:
            sorted_keys = sorted(parameters)  # Ensure deterministic o/p
            params_msg = ", ".join(f"{k}={parameters[k]}" for k in sorted_keys)
    if verbose > 9:
        start_msg = f"[CV{progress_msg}] START {params_msg}"
        print(f"{start_msg}{(80 - len(start_msg)) * '.'}")

    # Adjust length of sample weights
    fit_params = fit_params if fit_params is not None else {}
    fit_params = _check_fit_params(X, fit_params, train)

    if parameters is not None:
        # clone after setting parameters in case any parameters
        # are estimators (like pipeline steps)
        # because pipeline doesn't clone steps in fit
        cloned_parameters = {}
        for k, v in parameters.items():
            cloned_parameters[k] = clone(v, safe=False)

        estimator = estimator.set_params(**cloned_parameters)

    start_time = time.time()

    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, y_test = _safe_split(estimator, X, y, test, train)

    result = {}
    try:
        if y_train is None:
            estimator.fit(X_train, **fit_params)
        else:
            estimator.fit(X_train, y_train, **fit_params)

    except Exception:
        # Note fit time as time until error
        fit_time = time.time() - start_time
        score_time = 0.0
        if error_score == "raise":
            raise
        elif isinstance(error_score, numbers.Number):
            if isinstance(scorer, dict):
                test_scores = {name: error_score for name in scorer}
                if return_train_score:
                    train_scores = test_scores.copy()
            else:
                test_scores = error_score
                if return_train_score:
                    train_scores = error_score
        result["fit_error"] = format_exc()
    else:
        result["fit_error"] = None

        fit_time = time.time() - start_time
        train_scores = _score(estimator, X_train, y_train, scorer, error_score)
        test_scores = (
            _score(
                estimator,
                np.concat([X_train, X_test], axis=1),
                np.concat([y_train, y_test], axis=1),
                scorer,
                error_score,
            )
            - train_scores
        )
        score_time = time.time() - start_time - fit_time

    if verbose > 1:
        total_time = score_time + fit_time
        end_msg = f"[CV{progress_msg}] END "
        result_msg = params_msg + (";" if params_msg else "")
        if verbose > 2:
            if isinstance(test_scores, dict):
                for scorer_name in sorted(test_scores):
                    result_msg += f" {scorer_name}: ("
                    if return_train_score:
                        scorer_scores = train_scores[scorer_name]
                        result_msg += f"train={scorer_scores:.3f}, "
                    result_msg += f"test={test_scores[scorer_name]:.3f})"
            else:
                result_msg += ", score="
                if return_train_score:
                    result_msg += f"(train={train_scores:.3f}, test={test_scores:.3f})"
                else:
                    result_msg += f"{test_scores:.3f}"
        result_msg += f" total time={logger.short_format_time(total_time)}"

        # Right align the result_msg
        end_msg += "." * (80 - len(end_msg) - len(result_msg))
        end_msg += result_msg
        print(end_msg)

    result["test_scores"] = test_scores
    if return_train_score:
        result["train_scores"] = train_scores
    if return_n_test_samples:
        result["n_test_samples"] = _num_samples(X_test)
    if return_times:
        result["fit_time"] = fit_time
        result["score_time"] = score_time
    if return_parameters:
        result["parameters"] = parameters
    if return_estimator:
        result["estimator"] = estimator
    return result


@contextmanager
def _cache_net_forward_iter(net, use_caching, y_preds):
    """Caching context for ``skorch.NeuralNet`` instance.
    Returns a modified version of the net whose ``forward_iter``
    method will subsequently return cached predictions. Leaving the
    context will undo the overwrite of the ``forward_iter`` method.
    """
    if not use_caching:
        yield net
        return
    y_preds = iter(y_preds)

    # pylint: disable=unused-argument
    def cached_forward_iter(*args, device=net.device, **kwargs):
        for yp in y_preds:
            yield to_device(yp, device=device)

    net.forward_iter = cached_forward_iter
    try:
        yield net
    finally:
        # By setting net.forward_iter we define an attribute
        # `forward_iter` that precedes the bound method
        # `forward_iter`. By deleting the entry from the attribute
        # dict we undo this.
        del net.__dict__["forward_iter"]


class SurvivalEpochScoring(EpochScoring):
    def get_test_data(self, dataset_train, dataset_valid):
        """Return data needed to perform scoring.
        This is a convenience method that handles picking of
        train/valid, different types of input data, use of cache,
        etc. for you.
        Parameters
        ----------
        dataset_train
          Incoming training data or dataset.
        dataset_valid
          Incoming validation data or dataset.
        Returns
        -------
        X_test
          Input data used for making the prediction.
        y_test
          Target ground truth. If caching was enabled, return cached
          y_test.
        y_pred : list
          The predicted targets. If caching was disabled, the list is
          empty. If caching was enabled, the list contains the batches
          of the predictions. It may thus be necessary to concatenate
          the output before working with it:
          ``y_pred = np.concatenate(y_pred)``
        """
        dataset = dataset_valid

        if self.use_caching:
            X_test = dataset
            y_pred = self.y_preds_
            y_test = [self.target_extractor(y) for y in self.y_trues_]
            # In case of y=None we will not have gathered any samples.
            # We expect the scoring function to deal with y_test=None.
            y_test = np.concatenate(y_test) if y_test else None
            return X_test, y_test, y_pred

        if is_skorch_dataset(dataset):
            X_test, y_test = data_from_dataset(
                dataset,
                X_indexing=self.X_indexing_,
                y_indexing=self.y_indexing_,
            )
        else:
            X_test, y_test = dataset, None

        if y_test is not None:
            # We allow y_test to be None but the scoring function has
            # to be able to deal with it (i.e. called without y_test).
            y_test = self.target_extractor(y_test)
        return X_test, y_test, []

    def get_train_data(self, dataset_train, dataset_valid):
        """Return data needed to perform scoring.
        This is a convenience method that handles picking of
        train/valid, different types of input data, use of cache,
        etc. for you.
        Parameters
        ----------
        dataset_train
          Incoming training data or dataset.
        dataset_valid
          Incoming validation data or dataset.
        Returns
        -------
        X_test
          Input data used for making the prediction.
        y_test
          Target ground truth. If caching was enabled, return cached
          y_test.
        y_pred : list
          The predicted targets. If caching was disabled, the list is
          empty. If caching was enabled, the list contains the batches
          of the predictions. It may thus be necessary to concatenate
          the output before working with it:
          ``y_pred = np.concatenate(y_pred)``
        """
        dataset = dataset_train

        if self.use_caching:
            X_test = dataset
            y_pred = self.y_preds_
            y_test = [self.target_extractor(y) for y in self.y_trues_]
            # In case of y=None we will not have gathered any samples.
            # We expect the scoring function to deal with y_test=None.
            y_test = np.concatenate(y_test) if y_test else None
            return X_test, y_test, y_pred

        if is_skorch_dataset(dataset):
            X_test, y_test = data_from_dataset(
                dataset,
                X_indexing=self.X_indexing_,
                y_indexing=self.y_indexing_,
            )
        else:
            X_test, y_test = dataset, None

        if y_test is not None:
            # We allow y_test to be None but the scoring function has
            # to be able to deal with it (i.e. called without y_test).
            y_test = self.target_extractor(y_test)
        return X_test, y_test, []

    # pylint: disable=unused-argument,arguments-differ
    def on_epoch_end(self, net, dataset_train, dataset_valid, **kwargs):
        X_test, y_test, y_pred = self.get_test_data(dataset_train, dataset_valid)
        X_train, y_train, y_train_pred = self.get_train_data(
            dataset_train, dataset_valid
        )
        if X_test is None:
            raise ValueError

        with _cache_net_forward_iter(net, self.use_caching, y_pred) as cached_net:
            train_score = self._scoring(cached_net, X_test, y_test)
            joint_score = self._scoring(
                cached_net,
                torch.concat([X_train, X_test], axis=1),
                torch.concat([y_train, y_test], axis=1),
            )
            current_score = joint_score - train_score

        self._record_score(net.history, current_score)


# Adapted from https://github.com/pytorch/pytorch/issues/7068.
def seed_torch(seed=42):
    """Sets all seeds within torch and adjacent libraries.

    Args:
        seed: Random seed to be used by the seeding functions.

    Returns:
        None
    """
    # random.seed(seed)
    # os.environ["PYTHONHASHSEED"] = str(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return None


def transform_survival_target(time, event):
    return np.array([f"{time[i]}|{event[i]}" for i in range(len(time))])


def inverse_transform_survival_target(y):
    return (
        np.array([int(i.rsplit("|")[0]) for i in y]),
        np.array([int(i.rsplit("|")[1]) for i in y]),
    )


def create_risk_matrix(observed_survival_time):
    observed_survival_time = observed_survival_time.squeeze()
    return (
        (
            torch.outer(observed_survival_time, observed_survival_time)
            >= torch.square(observed_survival_time)
        )
        .long()
        .T
    )


def negative_partial_log_likelihood(
    predicted_log_hazard_ratio,
    observed_survival_time,
    observed_event_indicator,
):
    # print(observed_survival_time)
    if torch.sum(observed_event_indicator) <= 0.0:
        return torch.tensor(0.0, requires_grad=True)
    risk_matrix = create_risk_matrix(observed_survival_time)
    loss = -torch.sum(
        observed_event_indicator.float().squeeze()
        * (
            predicted_log_hazard_ratio.squeeze()
            - torch.log(
                torch.sum(
                    risk_matrix.float()
                    * torch.exp(predicted_log_hazard_ratio.squeeze()),
                    axis=1,
                )
            )
        )
    ) / torch.sum(observed_event_indicator)
    #print(predicted_log_hazard_ratio)
    if torch.isnan(loss) or torch.isinf(loss):
        #print("THIS IS IT")
        #print(loss)
        #print(predicted_log_hazard_ratio)
        #print(observed_survival_time)
        #print(observed_event_indicator)
        raise ValueError
    return loss


def transform_torch(time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
    """Transforms time, event into XGBoost digestable format.

    Parameters
    ----------
    time : npt.NDArray[float]
        Survival time.
    event : npt.NDArray[int]
        Boolean event indicator. Zero value is taken as censored event.

    Returns
    -------
    y : npt.NDArray[float]
        Transformed array containing survival time and event where negative value is taken as censored event.
    """
    # if isinstance(time, pd.Series):
    #    time = time.to_numpy()
    #    event = event.to_numpy()
    event_mod = event.clone()
    event_mod[event_mod == 0] = -1
    if (time == 0).any():
        raise RuntimeError("Data contains zero time value!")
        # alternative: time[time==0] = np.finfo(float).eps
    y = event_mod * time
    return y.float()


def transform_back_torch(y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Transforms XGBoost digestable format variable y into time and event.

    Parameters
    ----------
    y : npt.NDArray[float]
        Array containing survival time and event where negative value is taken as censored event.

    Returns
    -------
    tuple[npt.NDArray[float],npt.NDArray[int]]
        Survival time and event.
    """
    time = torch.abs(y)
    event = (torch.abs(y) == y).float()
    return time, event


# same using rv in pytorch
def eh_likelihood_torch_2(
    # y and linear predictor contain two cols
    y: torch.tensor,
    linear_predictor: torch.tensor,
) -> torch.tensor:
    if isinstance(linear_predictor, np.ndarray):
        linear_predictor = torch.from_numpy(linear_predictor)

    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y)
    sample_weight = 1.0
    bandwidth = None
    if isinstance(linear_predictor, np.ndarray):
        linear_predictor = torch.from_numpy(linear_predictor)

    time = torch.abs(y)
    event = (torch.abs(y) == y).float()
    # y1 = y[:,0]
    # time, event = transform_back_torch(y1)
    # need two predictors here
    linear_predictor_1: torch.tensor = torch.clamp(
        input=linear_predictor[:, 0] * sample_weight, min=-75, max=75
    )
    linear_predictor_2: torch.tensor = torch.clamp(
        input=linear_predictor[:, 1] * sample_weight, min=-75, max=75
    )
    exp_linear_predictor_1 = torch.exp(linear_predictor_1) + 1e-10
    exp_linear_predictor_2 = torch.exp(linear_predictor_2) + 1e-10

    n_events: int = torch.sum(event)
    n_samples: int = time.shape[0]
    if bandwidth is None:
        bandwidth = 1.30 * torch.pow(n_samples, torch.tensor(-0.2))
    R_linear_predictor: torch.tensor = torch.log(time * exp_linear_predictor_1)
    if torch.any(torch.isnan(R_linear_predictor)) or torch.any(
        torch.isinf(R_linear_predictor)
    ):
        #print(linear_predictor_1.shape)
        #print(linear_predictor)
        #print(exp_linear_predictor_1)
        #print(R_linear_predictor)
        #print(time)
        #print(event)

        raise ValueError
    inverse_sample_size_bandwidth: float = 1 / (n_samples * bandwidth)
    event_mask: torch.tensor = event.bool()

    _: torch.tensor
    kernel_matrix: torch.tensor
    integrated_kernel_matrix: torch.tensor

    inverse_sample_size_bandwidth: float = 1 / (n_samples * bandwidth)
    event_mask = event.bool()
    rv = torch.distributions.normal.Normal(0, 1, validate_args=None)
    # Cast to double to prevent issues with precision when dividing
    # the exped linear predictors.
    sample_repeated_linear_predictor = (
        (exp_linear_predictor_2.double() / exp_linear_predictor_1.double())
        .repeat((int(n_events.item()), 1))
        .T
    )
    diff = (
        R_linear_predictor.reshape(-1, 1) - R_linear_predictor[event_mask]
    ) / bandwidth

    kernel_matrix = torch.exp(-1 / 2 * torch.square(diff[event_mask, :])) / torch.sqrt(
        torch.tensor(2) * torch.pi
    )
    integrated_kernel_matrix = rv.cdf(diff)

    inverse_sample_size: float = 1 / n_samples
    kernel_sum = kernel_matrix.sum(axis=0)
    integrated_kernel_sum = (
        sample_repeated_linear_predictor * integrated_kernel_matrix
    ).sum(axis=0)
    likelihood: torch.tensor = inverse_sample_size * (
        linear_predictor_2[event_mask].sum()
        - R_linear_predictor[event_mask].sum()
        + torch.log(inverse_sample_size_bandwidth * kernel_sum).sum()
        - torch.log(inverse_sample_size * integrated_kernel_sum).sum()
    )
    if torch.isnan(likelihood) or torch.isinf(likelihood):
        # print(likelihood)
        # print(linear_predictor_2[event_mask])
        # print(R_linear_predictor[event_mask])
        # print(inverse_sample_size_bandwidth * kernel_sum)
        # print(diff)
        # print(exp_linear_predictor_2)
        # print(exp_linear_predictor_1)
        # print(inverse_sample_size * integrated_kernel_sum)

        # print(y)
        # print(linear_predictor_1)
        # print(linear_predictor_2)
        raise ValueError
    return -likelihood


PDF_PREFACTOR: float = 0.3989424488876037
SQRT_TWO: float = 1.4142135623730951
SQRT_EPS: float = 1.4901161193847656e-08
EPS: float = 2.220446049250313e-16
CDF_ZERO: float = 0.5


# @jit(nopython=True, cache=True, fastmath=True)
def bandwidth_function(time, event, n):
    return (8 * (math.sqrt(2) / 3)) ** (1 / 5) * n ** (-1 / 5)


# @jit(nopython=True, cache=True, fastmath=True)
def gaussian_integrated_kernel(x):
    return 0.5 * (1 + math.erf(x / SQRT_TWO))


# @jit(nopython=True, cache=True, fastmath=True)
def gaussian_kernel(x):
    return PDF_PREFACTOR * math.exp(-0.5 * (x**2))


@jit(nopython=True, cache=True, fastmath=True)
def kernel(a, b, bandwidth):
    kernel_matrix: np.array = np.empty(shape=(a.shape[0], b.shape[0]))
    for ix in range(a.shape[0]):
        for qx in range(b.shape[0]):
            kernel_matrix[ix, qx] = gaussian_kernel((a[ix] - b[qx]) / bandwidth)
    return kernel_matrix


@jit(nopython=True, cache=True, fastmath=True)
def integrated_kernel(a, b, bandwidth):
    integrated_kernel_matrix: np.array = np.empty(shape=(a.shape[0], b.shape[0]))
    for ix in range(a.shape[0]):
        for qx in range(b.shape[0]):
            integrated_kernel_matrix[ix, qx] = gaussian_integrated_kernel(
                (a[ix] - b[qx]) / bandwidth
            )
    return integrated_kernel_matrix


# this is key
# @jit(nopython=True, cache=True, fastmath=True)
def baseline_hazard_estimator_eh(
    time: np.array,
    time_train: np.array,
    event_train: np.array,
    predictor_train: np.array,
):
    """Get Extended Hazard Model's baseline hazard.

    Parameters
    ----------
    time : np.array
        _description_
    time_train : np.array
        _description_
    event_train : np.array
        _description_
    predictor_train : np.array
        _description_

    Returns
    -------
    np.array
        Baseline hazard.
    """
    n_samples: int = time_train.shape[0]
    bandwidth = 1.30 * math.pow(n_samples, -0.2)
    inverse_bandwidth: float = 1 / bandwidth
    inverse_sample_size: float = 1 / n_samples
    inverse_bandwidth_sample_size: float = (
        inverse_sample_size * (1 / (time + EPS)) * inverse_bandwidth
    )
    log_time: float = np.log(time + EPS)
    train_eta_1 = predictor_train[:, 0]
    train_eta_2 = predictor_train[:, 1]
    h_prefactor = np.exp(np.clip(train_eta_2 - train_eta_1, -75, 75))
    if np.any(np.isnan(h_prefactor)) or np.any(np.isinf(h_prefactor)):
        print(train_eta_2)
        print(train_eta_1)
        print(h_prefactor)
        raise ValueError
    R_lp: np.array = np.log(time_train * np.exp(predictor_train[:, 0]))
    difference_lp_log_time: np.array = (R_lp - log_time) / bandwidth
    numerator: float = 0.0
    denominator: float = 0.0
    for _ in range(n_samples):
        difference: float = difference_lp_log_time[_]

        denominator += h_prefactor[_] * gaussian_integrated_kernel(difference)
        if event_train[_]:
            numerator += gaussian_kernel(difference)
    numerator = inverse_bandwidth_sample_size * numerator
    denominator = inverse_sample_size * denominator

    if denominator <= 0.0:
        return 0.0
    else:
        return numerator / denominator


def transform(time, event):
    """Transforms time, event into XGBoost digestable format.

    Parameters
    ----------
    time : npt.NDArray[float]
        Survival time.
    event : npt.NDArray[int]
        Boolean event indicator. Zero value is taken as censored event.

    Returns
    -------
    y : npt.NDArray[float]
        Transformed array containing survival time and event where negative value is taken as censored event.
    """
    # if isinstance(time, pd.Series):
    #    time = time.to_numpy()
    #    event = event.to_numpy()
    event_mod = np.copy(event)
    event_mod[event_mod == 0] = -1
    if np.any(time == 0):
        raise RuntimeError("Data contains zero time value!")
        # alternative: time[time==0] = np.finfo(float).eps
    y = event_mod * time
    return y


def transform_back(y):
    """Transforms XGBoost digestable format variable y into time and event.

    Parameters
    ----------
    y : npt.NDArray[float]
        Array containing survival time and event where negative value is taken as censored event.

    Returns
    -------
    tuple[npt.NDArray[float],npt.NDArray[int]]
        Survival time and event.
    """
    time = np.abs(y)
    event = np.abs(y) == y
    event = event.astype(np.int64)  # for numba
    return time, event


def get_cumulative_hazard_function_eh(
    X_train: np.array,
    X_test: np.array,
    y_train: np.array,
    y_test: np.array,
    predictor_train: np.array,
    predictor_test: np.array,
) -> pd.DataFrame:
    """Get cumulative hazard function of Extended Hazards Model.

    Parameters
    ----------
    X_train : np.array
        _description_
    X_test : np.array
        _description_
    y_train : np.array
        _description_
    y_test : np.array
        _description_
    predictor_train : np.array
        _description_
    predictor_test : np.array
        _description_

    Returns
    -------
    pd.DataFrame
        Cumulative hazard dataframe.
    """

    # if y_test.ndim==2:
    #    time_test, event_test = transform_back(y_test[:,0])
    # else:
    #    time_test, event_test = transform_back(y_test)

    # if y_train.ndim==2:
    #    time_train, event_train = transform_back(y_train[:,0])
    # else:
    #    time_train, event_train = transform_back(y_train)
    time_train = np.abs(y_train)
    event_train = (np.abs(y_train) == y_train).astype(int)

    time: np.array = np.unique(time_train)
    predictor_train = np.clip(a=predictor_train, a_min=-75, a_max=75)
    predictor_test = np.clip(a=predictor_test, a_min=-75, a_max=75)

    theta: np.array = np.exp(predictor_test)
    n_samples: int = predictor_test.shape[0]
    zero_flag: bool = False
    if 0 not in time:
        zero_flag = True
        time = np.concatenate([np.array([0]), time])
        cumulative_hazard: np.array = np.empty((n_samples, time.shape[0]))
    else:
        cumulative_hazard: np.array = np.empty((n_samples, time.shape[0]))

    def hazard_function_integrate(s):
        return baseline_hazard_estimator_eh(
            time=s,
            time_train=time_train,
            event_train=event_train,
            predictor_train=predictor_train,
        )

    integration_times = np.stack(
        [
            np.unique(
                np.ravel(y_train)[
                    np.ravel((np.abs(y_train) == y_train).astype(np.bool_))
                ]
            )
            * i
            for i in np.round(np.exp(np.ravel(predictor_test)), 2)
        ]
    )
    integration_times = np.unique((np.ravel(integration_times)))

    integration_times = np.concatenate(
        [[0], integration_times, [np.max(integration_times) + 0.01]]
    )
    # print(integration_times.shape)
    integration_values = np.zeros(integration_times.shape[0])
    for _ in range(1, integration_values.shape[0]):
        integration_values[_] = (
            integration_values[_ - 1]
            + quadrature(
                func=hazard_function_integrate,
                a=integration_times[_ - 1],
                b=integration_times[_],
                vec_func=False,
            )[0]
        )

    for _ in range(n_samples):
        cumulative_hazard[_] = (
            integration_values[
                np.digitize(x=time * theta[_, 0], bins=integration_times, right=False)
                - 1
            ]
            * theta[_, 1]
            / theta[_, 0]
        )
    if zero_flag:
        cumulative_hazard = cumulative_hazard[:, 1:]
        time = time[1:]
    return pd.DataFrame(cumulative_hazard, columns=time)


def breslow_likelihood_torch(
    y: torch.Tensor, log_partial_hazard: torch.Tensor
) -> torch.Tensor:
    """Generate negative loglikelihood (loss) according to Breslow.
    Assumes times have been sorted beforehand.

    Parameters
    ----------
    y : npt.NDArray[float]
        Sorted array containing survival time and event where negative value is taken as censored event.
    log_partial_hazard : npt.NDArray[float]
        Estimated hazard.

    Returns
    -------
    npt.NDArray[float]
        Negative loglikelihood (loss) according to Breslow.
    """

    if isinstance(log_partial_hazard, np.ndarray):
        log_partial_hazard = torch.from_numpy(log_partial_hazard)

    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y)
    log_partial_hazard = log_partial_hazard.double()
    time = torch.abs(y)
    event = (torch.abs(y) == y).float()
    log_partial_hazard = torch.clamp(log_partial_hazard, -75, 75)
    # Assumes times have been sorted beforehand.
    partial_hazard = torch.exp(log_partial_hazard)
    n_events = torch.sum(event)
    n_samples = time.shape[0]
    previous_time = time[0]
    risk_set_sum = 0
    likelihood = 0
    set_count = 0
    accumulated_sum = 0
    risk_set_sum = torch.sum(partial_hazard)

    for k in range(n_samples):
        current_time = time[k]
        if current_time > previous_time:
            if set_count > 0:
                likelihood = likelihood - (set_count * torch.log(risk_set_sum))
            risk_set_sum = risk_set_sum - accumulated_sum
            set_count = 0
            accumulated_sum = 0

        if event[k]:
            set_count = set_count + 1
            likelihood = likelihood + log_partial_hazard[k]

        previous_time = current_time
        accumulated_sum = accumulated_sum + partial_hazard[k]
    if set_count > 0:
        likelihood = likelihood - (set_count * torch.log(risk_set_sum))
    if torch.isnan(likelihood) or torch.isinf(likelihood):
        #print(likelihood)
        #print(event)
        #print(time)
        #print(log_partial_hazard)
        raise ValueError
    return -likelihood / n_events


class BreslowLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = "mean") -> None:
        super().__init__(size_average, reduce, reduction)
        # Initialize any additional variables you need for your custom loss function here.

    def forward(self, prediction, input):
        time, event = transform_back_torch(input)
        #print(prediction)
        loss = negative_partial_log_likelihood(prediction, time, event)
        return loss


class EHLoss(_Loss):
    def __init__(
        self, size_average=None, reduce=None, reduction: str = "mean", bandwidth=None
    ) -> None:
        super().__init__(size_average, reduce, reduction)
        # Initialize any additional variables you need for your custom loss function here.
        self.bandwidth = bandwidth

    def forward(self, prediction, input):
        loss = eh_likelihood_torch_2(input, prediction)
        return loss


def get_noise_modalities(n_noise_modalities, time, noise_modality_dimensionality):
    noise_modalities = [
        np.random.normal(size=(time.shape[0], noise_modality_dimensionality))
        for ix in range(n_noise_modalities)
    ]
    # if noised_target:
    #    noise_modalities[0] = np.expand_dims(
    #        time.to_numpy(), 1
    #    ) + np.random.standard_normal(np.expand_dims(time.to_numpy(), 1).shape)
    for ix in range(len(noise_modalities)):
        noise_modalities[ix] = pd.DataFrame(
            noise_modalities[ix],
            columns=[
                f"noise-{ix}_feature_{jx}"
                for jx in range(noise_modalities[ix].shape[1])
            ],
        )
    return noise_modalities


def calculate_log_hazard_input_size(fusion_method, blocks, modality_dimension):
    match fusion_method:
        case "early":
            return sum([len(block) for block in blocks])
        case "early_ae":
            return modality_dimension
        case "late_mean":
            raise ValueError
        case "late_moe":
            raise ValueError
        case "intermediate_mean":
            return modality_dimension
        case "intermediate_max":
            return modality_dimension
        case "intermediate_concat":
            return modality_dimension * len(blocks)
        case "intermediate_ae":
            return modality_dimension * len(blocks)
        case "intermediate_embrace":
            return modality_dimension
        case "intermediate_attention":
            return modality_dimension


def drop_constants(data):
    mask = (data != data.iloc[0]).any()
    return data.loc[:, mask]


def filter_modalities(data, selected_modalities_ix, all_modalities):
    modalities_to_keep_ix = np.array(
        [int(i) for i in selected_modalities_ix.rsplit(",")]
    )
    all_modalities = np.array(all_modalities)
    modalities_to_keep = all_modalities[modalities_to_keep_ix]
    modality_mask = [
        col for col in data.columns if col.rsplit("_")[0] in modalities_to_keep
    ]
    return data[modality_mask]


class StratifiedSkorchSurvivalSplit(ValidSplit):
    """Adapt `ValidSplit` to make it usable with our adapted
    survival target string format.

    For further documentation, please refer to the `ValidSplit`
    documentation, as the only changes made were to adapt the string
    target format.
    """

    def __call__(self, dataset, y=None, groups=None):
        if y is not None:
            # Handle string target by selecting out only the event
            # to stratify on.
            # if y.dtype not in [np.dtype("float32"), np.dtype("int")]:
            if np.min(np.min(y) < 0.0):
                # y = np.array([str.rsplit(i, "|")[1] for i in y]).astype(
                #    np.float32
                # )
                y = (np.abs(y) == y).astype(int)

        bad_y_error = ValueError(
            "Stratified CV requires explicitly passing a suitable y."
        )

        if (y is None) and self.stratified:
            raise bad_y_error

        # cv = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        cv = self.check_cv(y)
        # pylint: disable=invalid-name
        len_dataset = get_len(dataset)
        if y is not None:
            len_y = get_len(y)
            if len_dataset != len_y:
                raise ValueError(
                    "Cannot perform a CV split if dataset and y "
                    "have different lengths."
                )

        args = (np.arange(len_dataset),)
        if self._is_stratified(cv):
            args = args + (to_numpy(y),)

        idx_train, idx_valid = next(iter(cv.split(*args, groups=groups)))
        dataset_train = torch.utils.data.Subset(dataset, idx_train)
        dataset_valid = torch.utils.data.Subset(dataset, idx_valid)
        return dataset_train, dataset_valid


def get_blocks(feature_names):
    column_types = pd.Series(feature_names).str.rsplit("_").apply(lambda x: x[0]).values
    return [
        np.where(
            modality
            == pd.Series(feature_names).str.rsplit("_").apply(lambda x: x[0]).values
        )[0].tolist()
        for modality in [
            q
            for q in [
                "clinical",
                "gex",
                "rppa",
                "mirna",
                "mut",
                "meth",
                "cnv",
            ]
            + [f"noise-{ix}" for ix in range(10)]
            if q in np.unique(column_types)
        ]
    ]


class StratifiedSurvivalKFold(StratifiedKFold):
    """Adapt `StratifiedKFold` to make it usable with our adapted
    survival target string format.

    For further documentation, please refer to the `StratifiedKFold`
    documentation, as the only changes made were to adapt the string
    target format.
    """

    def _make_test_folds(self, X, y=None):
        if y is not None and isinstance(y, np.ndarray):
            # Handle string target by selecting out only the event
            # to stratify on.
            if np.min(np.min(y) < 0.0):
                # print(y)
                # y = np.array([str.rsplit(i, "|")[1] for i in y]).astype(
                #    np.float32
                # )
                y = (np.abs(y) == y).astype(np.float32)

        return super()._make_test_folds(X=X, y=y)

    def _iter_test_masks(self, X, y=None, groups=None):
        if y is not None and isinstance(y, np.ndarray):
            # Handle string target by selecting out only the event
            # to stratify on.

            if np.min(np.min(y) < 0.0):
                # print(y)
                # y = np.array([str.rsplit(i, "|")[1] for i in y]).astype(
                #    np.float32
                # )
                y = (np.abs(y) == y).astype(np.float32)
                # return super()._iter_test_masks(X, y=event)
        return super()._iter_test_masks(X, y=y)

    def split(self, X, y, groups=None):
        return super().split(X=X, y=y, groups=groups)


def negative_partial_log_likelihood_loss(
    y_true,
    y_pred,
):
    # print("-----")
    # print("TRUE")
    # print(y_true.shape)
    # print("PRED")
    # print(y_pred.shape)
    # print("------")
    # print(y_true)
    # raise ValueError
    (
        observed_survival_time,
        observed_event_indicator,
    ) = transform_back(y_true)
    return negative_partial_log_likelihood(
        y_pred,
        torch.tensor(observed_survival_time),
        torch.tensor(observed_event_indicator),
    )


def negative_partial_log_likelihood_loss_multi_output(
    y_true,
    y_pred,
):
    (
        observed_survival_time,
        observed_event_indicator,
    ) = inverse_transform_survival_target(y_true)
    return negative_partial_log_likelihood(
        y_pred[0],
        torch.tensor(observed_survival_time),
        torch.tensor(observed_event_indicator),
    )


class RepeatedStratifiedSurvivalKFold(_RepeatedSplits):
    def __init__(self, *, n_splits=5, n_repeats=10, random_state=None):
        super().__init__(
            StratifiedSurvivalKFold,
            n_repeats=n_repeats,
            random_state=random_state,
            n_splits=n_splits,
        )
