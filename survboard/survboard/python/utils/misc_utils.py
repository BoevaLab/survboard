import math
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from numba import jit
from scipy.integrate import quadrature
from sklearn.model_selection import StratifiedKFold
from skorch.dataset import ValidSplit, get_len
from skorch.utils import to_numpy
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


# Adapted from https://github.com/pytorch/pytorch/issues/7068.
def seed_torch(seed=42):
    """Sets all seeds within torch and adjacent libraries.

    Args:
        seed: Random seed to be used by the seeding functions.

    Returns:
        None
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return None


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
    if torch.isnan(loss) or torch.isinf(loss):
        raise ValueError
    return loss


def transform_back_torch(y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    time = torch.abs(y)
    event = (torch.abs(y) == y).float()
    return time, event


def eh_likelihood_torch_2(
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
        raise ValueError
    return -likelihood


PDF_PREFACTOR: float = 0.3989424488876037
SQRT_TWO: float = 1.4142135623730951
SQRT_EPS: float = 1.4901161193847656e-08
EPS: float = 2.220446049250313e-16
CDF_ZERO: float = 0.5


def bandwidth_function(time, event, n):
    return (8 * (math.sqrt(2) / 3)) ** (1 / 5) * n ** (-1 / 5)


def gaussian_integrated_kernel(x):
    return 0.5 * (1 + math.erf(x / SQRT_TWO))


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
    """
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
    event_mod = np.copy(event)
    event_mod[event_mod == 0] = -1
    # if np.any(time == 0):
    #     raise RuntimeError("Data contains zero time value!")
    y = event_mod * time
    return y


def transform_back(y):
    """
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
    event = event.astype(np.int64)
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


class BreslowLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = "mean") -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self, prediction, input):
        time, event = transform_back_torch(input)
        loss = negative_partial_log_likelihood(prediction, time, event)
        return loss


class EHLoss(_Loss):
    def __init__(
        self, size_average=None, reduce=None, reduction: str = "mean", bandwidth=None
    ) -> None:
        super().__init__(size_average, reduce, reduction)
        self.bandwidth = bandwidth

    def forward(self, prediction, input):
        loss = eh_likelihood_torch_2(input, prediction)
        return loss


class DiscreteTimeLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = "mean") -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self, prediction, input):
        time, event = transform_back_torch(input)
        # event[time == 0] = 0
        loss = nll_logistic_hazard(prediction, time, event)
        return loss


def sparse_group_lasso(weights, lamb, alpha, groups):
    lasso_penalty = torch.norm(input=weights, p=1)
    group_lasso_penalty = 0
    for group in groups:
        group_lasso_penalty += torch.sqrt(torch.tensor(len(group))) * torch.norm(
            input=weights[:, group], p=2
        )

    # print("Lasso")
    # print(lamb * alpha * lasso_penalty)
    # print("Group Lasso")
    # print(lamb * (1 - alpha) * group_lasso_penalty)
    return lamb * alpha * lasso_penalty + lamb * (1 - alpha) * group_lasso_penalty


class BreslowSGLLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = "mean") -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self, prediction, input, alpha, lamb, weights, groups):
        time, event = transform_back_torch(input)
        # print("Loss")
        # print(negative_partial_log_likelihood(prediction, time, event) )
        # print(sparse_group_lasso(weights=weights, lamb=lamb, alpha=alpha, groups=groups))
        loss = negative_partial_log_likelihood(
            prediction, time, event
        ) + sparse_group_lasso(weights=weights, lamb=lamb, alpha=alpha, groups=groups)
        return loss


class EHSGLLoss(_Loss):
    def __init__(
        self, size_average=None, reduce=None, reduction: str = "mean", bandwidth=None
    ) -> None:
        super().__init__(size_average, reduce, reduction)
        self.bandwidth = bandwidth

    def forward(self, prediction, input, alpha, lamb, weights, groups):
        loss = eh_likelihood_torch_2(input, prediction) + sparse_group_lasso(
            weights=weights, lamb=lamb, alpha=alpha, groups=groups
        )
        return loss


class DiscreteTimeSGLLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = "mean") -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self, prediction, input, alpha, lamb, weights, groups):
        time, event = transform_back_torch(input)
        loss = nll_logistic_hazard(prediction, time, event) + sparse_group_lasso(
            weights=weights, lamb=lamb, alpha=alpha, groups=groups
        )
        return loss


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


class StratifiedSkorchSurvivalSplit(ValidSplit):
    """Adapt `ValidSplit` to make it usable with our adapted
    survival target string format.

    For further documentation, please refer to the `ValidSplit`
    documentation, as the only changes made were to adapt the string
    target format.
    """

    def __call__(self, dataset, y=None, groups=None):
        if y is not None:
            if np.min(np.min(y) < 0.0):
                y = (np.abs(y) == y).astype(int)

        bad_y_error = ValueError(
            "Stratified CV requires explicitly passing a suitable y."
        )

        if (y is None) and self.stratified:
            raise bad_y_error

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
                y = (np.abs(y) == y).astype(np.float32)

        return super()._make_test_folds(X=X, y=y)

    def _iter_test_masks(self, X, y=None, groups=None):
        if y is not None and isinstance(y, np.ndarray):
            # Handle string target by selecting out only the event
            # to stratify on.

            if np.min(np.min(y) < 0.0):
                y = (np.abs(y) == y).astype(np.float32)
        return super()._iter_test_masks(X, y=y)

    def split(self, X, y, groups=None):
        return super().split(X=X, y=y, groups=groups)


def negative_partial_log_likelihood_loss(
    y_true,
    y_pred,
):
    (
        observed_survival_time,
        observed_event_indicator,
    ) = transform_back(y_true)
    return negative_partial_log_likelihood(
        y_pred,
        torch.tensor(observed_survival_time),
        torch.tensor(observed_event_indicator),
    )


# Adapated from https://github.com/havakv/pycox/blob/master/pycox/models/loss.py
def nll_logistic_hazard_loss(y_true, y_pred):
    (
        observed_survival_time,
        observed_event_indicator,
    ) = transform_back(y_true)
    return nll_logistic_hazard(
        y_pred,
        torch.tensor(observed_survival_time),
        torch.tensor(observed_event_indicator),
    )


# Adapated from https://github.com/havakv/pycox/blob/master/pycox/models/loss.py
def nll_logistic_hazard(phi, idx_durations, observed_event_indicator):
    observed_event_indicator[idx_durations == 0] = 0
    idx_durations = idx_durations.long()
    observed_event_indicator = observed_event_indicator.long()
    observed_event_indicator = observed_event_indicator.view(-1, 1)
    idx_durations = idx_durations.view(-1, 1)
    y_bce = (
        torch.zeros_like(phi)
        .long()
        .scatter(1, idx_durations, observed_event_indicator)
        .float()
    )
    bce = F.binary_cross_entropy_with_logits(phi, y_bce, reduction="none")
    loss = (
        bce.cumsum(1).gather(1, idx_durations).view(-1).sum() / idx_durations.shape[0]
    )
    return loss


# From: https://github.com/havakv/pycox/blob/master/pycox/models/utils.py
def pad_col(input, val=0, where="end"):
    if len(input.shape) != 2:
        raise ValueError(f"Only works for `phi` tensor that is 2-D.")
    pad = torch.zeros_like(input[:, :1])
    if val != 0:
        pad = pad + val
    if where == "end":
        return torch.cat([input, pad], dim=1)
    elif where == "start":
        return torch.cat([pad, input], dim=1)
    raise ValueError(f"Need `where` to be 'start' or 'end', got {where}")


# Adapted from: https://github.com/havakv/pycox/blob/master/pycox/models/utils.py
def make_subgrid(grid, sub=1):
    subgrid = np.concatenate(
        [
            np.concatenate(
                [
                    np.linspace(start, end, num=sub + 1)[:-1]
                    for start, end in zip(grid[:-1], grid[1:])
                ]
            ),
            np.array([grid[-1]]),
        ]
    )
    return subgrid


# Adapted from: https://github.com/havakv/pycox/blob/master/pycox/models/interpolation.py
class InterpolateDiscrete:
    def __init__(
        self, model, scheme="const_pdf", duration_index=None, sub=10, epsilon=1e-7
    ):
        self.model = model
        self.scheme = scheme
        self.duration_index = duration_index
        self.sub = sub
        self.epsilon = epsilon

    @property
    def sub(self):
        return self._sub

    @sub.setter
    def sub(self, sub):
        if type(sub) is not int:
            raise ValueError(f"Need `sub` to have type `int`, got {type(sub)}")
        self._sub = sub

    def predict_hazard(self, input):
        raise NotImplementedError

    def predict_pmf(self, input):
        raise NotImplementedError

    def predict_surv(self, input):
        return self._surv_const_pdf(input)

    def _surv_const_pdf(self, input):
        s = self.model.predict_surv(input)
        n, m = s.shape
        device = s.device
        diff = (
            (s[:, 1:] - s[:, :-1])
            .contiguous()
            .view(-1, 1)
            .repeat(1, self.sub)
            .view(n, -1)
        )
        rho = (
            torch.linspace(0, 1, self.sub + 1, device=device)[:-1]
            .contiguous()
            .repeat(n, m - 1)
        )
        s_prev = s[:, :-1].contiguous().view(-1, 1).repeat(1, self.sub).view(n, -1)
        surv = torch.zeros(n, int((m - 1) * self.sub + 1))
        surv[:, :-1] = diff * rho + s_prev
        surv[:, -1] = s[:, -1]
        return surv

    def predict_surv_df(self, input):
        surv = self.predict_surv(input)
        index = None
        if self.duration_index is not None:
            index = make_subgrid(self.duration_index, self.sub)
        return pd.DataFrame(surv, columns=index)


# Adapted from: https://github.com/havakv/pycox/blob/master/pycox/models/interpolation.py
class InterpolateLogisticHazard(InterpolateDiscrete):
    def predict_hazard(self, input):
        if self.scheme in ["const_hazard", "exp_surv"]:
            haz = self._hazard_const_haz(input)
        else:
            raise NotImplementedError
        return haz

    def predict_surv(self, input):
        if self.scheme in ["const_hazard", "exp_surv"]:
            surv = self._surv_const_haz(input)
        elif self.scheme in ["const_pdf", "lin_surv"]:
            surv = self._surv_const_pdf(input)
        else:
            raise NotImplementedError
        return surv

    def _hazard_const_haz(self, input):
        haz_orig = self.model.predict_hazard(input)
        haz = (1 - haz_orig).add(self.epsilon).log().mul(-1).relu()[:, 1:].contiguous()
        n = haz.shape[0]
        haz = haz.view(-1, 1).repeat(1, self.sub).view(n, -1).div(self.sub)
        haz = pad_col(haz, where="start")
        haz[:, 0] = haz_orig[:, 0]
        return haz

    def _surv_const_haz(self, input):
        haz = self._hazard_const_haz(input)
        surv_0 = 1 - haz[:, :1]
        surv = pad_col(haz[:, 1:], where="start").cumsum(1).mul(-1).exp().mul(surv_0)
        return surv


# Need in model:
# predict_surv
# predict_hazard


# All code below from: https://github.com/havakv/pycox/blob/master/pycox/preprocessing/label_transforms.py


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


def idx_at_times(index_surv, times, steps="pre", assert_sorted=True):
    """Gives index of `index_surv` corresponding to `time`, i.e.
    `index_surv[idx_at_times(index_surv, times)]` give the values of `index_surv`
    closet to `times`.

    Arguments:
        index_surv {np.array} -- Durations of survival estimates
        times {np.array} -- Values one want to match to `index_surv`

    Keyword Arguments:
        steps {str} -- Round 'pre' (closest value higher) or 'post'
          (closest value lower) (default: {'pre'})
        assert_sorted {bool} -- Assert that index_surv is monotone (default: {True})

    Returns:
        np.array -- Index of `index_surv` that is closest to `times`
    """
    if assert_sorted:
        assert pd.Series(
            index_surv
        ).is_monotonic_increasing, "Need 'index_surv' to be monotonic increasing"
    if steps == "pre":
        idx = np.searchsorted(index_surv, times)
    elif steps == "post":
        idx = np.searchsorted(index_surv, times, side="right") - 1
    return idx.clip(0, len(index_surv) - 1)


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


def cuts_quantiles(durations, events, num, min_=0.0, dtype="float64"):
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


def transform_discrete_time(time, event, n_durations, scheme="quantiles"):
    trans = LabTransDiscreteTime(cuts=n_durations, scheme=scheme)
    y = trans.fit_transform(time, event)
    return (y, trans.cuts)
