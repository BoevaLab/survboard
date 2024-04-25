import math

import numpy as np
import pandas as pd
import torch
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
    if np.any(time == 0):
        raise RuntimeError("Data contains zero time value!")
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
