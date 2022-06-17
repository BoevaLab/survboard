import torch
from skorch.net import NeuralNet
from skorch.utils import to_tensor

from survival_benchmark.python.breslow import BreslowEstimator
from survival_benchmark.python.utils.utils import inverse_transform_survival_target


class CoxPHNet(NeuralNet):
    def get_loss(self, y_pred, y_true, X=None, training=False):
        """Return the loss for this batch.
        Parameters
        ----------
        y_pred : torch tensor
          Predicted target values
        y_true : torch tensor
          True target values.
        X : input data, compatible with skorch.dataset.Dataset
          By default, you should be able to pass:
            * numpy arrays
            * torch tensors
            * pandas DataFrame or Series
            * scipy sparse CSR matrices
            * a dictionary of the former three
            * a list/tuple of the former three
            * a Dataset
          If this doesn't work with your data, you have to pass a
          ``Dataset`` that can deal with the data.
        training : bool (default=False)
          Whether train mode should be used or not.
        """
        time, event = inverse_transform_survival_target(y_true)
        time = to_tensor(time, device=self.device)
        event = to_tensor(event, device=self.device)
        y_true = torch.stack([time, event], axis=1)
        return self.criterion_(y_pred, y_true)

    def fit(self, X, y=None, **fit_params):
        if not self.warm_start or not self.initialized_:
            self.initialize()

        time, event = inverse_transform_survival_target(y)
        self.train_time = time
        self.train_event = event
        self.partial_fit(X, y, **fit_params)
        self.fit_breslow(
            self.module_.forward(torch.tensor(X)).detach().numpy().ravel(),
            time,
            event,
        )
        return self

    def fit_breslow(self, log_hazard_ratios, time, event):
        self.breslow = BreslowEstimator().fit(log_hazard_ratios, event, time)

    def predict_survival_function(self, X):
        log_hazard_ratios = self.forward(X)
        survival_function = self.breslow.get_survival_function(
            log_hazard_ratios
        )
        return survival_function

    def predict(self, X):
        log_hazard_ratios = self.forward(X)
        return log_hazard_ratios


class IntermediateFusionMeanNet(CoxPHNet):
    def get_loss(self, y_pred, y_true, X=None, training=False):
        """Return the loss for this batch.
        Parameters
        ----------
        y_pred : torch tensor
          Predicted target values
        y_true : torch tensor
          True target values.
        X : input data, compatible with skorch.dataset.Dataset
          By default, you should be able to pass:
            * numpy arrays
            * torch tensors
            * pandas DataFrame or Series
            * scipy sparse CSR matrices
            * a dictionary of the former three
            * a list/tuple of the former three
            * a Dataset
          If this doesn't work with your data, you have to pass a
          ``Dataset`` that can deal with the data.
        training : bool (default=False)
          Whether train mode should be used or not.
        """
        time, event = inverse_transform_survival_target(y_true)
        time = to_tensor(time, device=self.device)
        event = to_tensor(event, device=self.device)
        y_true = torch.stack([time, event], axis=1)
        return self.criterion_(y_pred, y_true, self.module_.alpha)

    def fit(self, X, y=None, **fit_params):
        if not self.warm_start or not self.initialized_:
            self.initialize()

        time, event = inverse_transform_survival_target(y)
        self.train_time = time
        self.train_event = event
        self.partial_fit(X, y, **fit_params)
        self.fit_breslow(
            self.module_.forward(torch.tensor(X))[0].detach().numpy().ravel(),
            time,
            event,
        )
        return self

    def forward(
        self,
        X,
        training=False,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        y_infer = list(self.forward_iter(X, training=training, device=device))
        if len(y_infer) > 1:
            y_infer = torch.cat([i[0] for i in y_infer], axis=0)
        else:
            y_infer = y_infer[0][0]
        return y_infer


class NaiveNeuralNet(CoxPHNet):
    def get_loss(self, y_pred, y_true, X=None, training=False):
        """Return the loss for this batch.
        Parameters
        ----------
        y_pred : torch tensor
          Predicted target values
        y_true : torch tensor
          True target values.
        X : input data, compatible with skorch.dataset.Dataset
          By default, you should be able to pass:
            * numpy arrays
            * torch tensors
            * pandas DataFrame or Series
            * scipy sparse CSR matrices
            * a dictionary of the former three
            * a list/tuple of the former three
            * a Dataset
          If this doesn't work with your data, you have to pass a
          ``Dataset`` that can deal with the data.
        training : bool (default=False)
          Whether train mode should be used or not.
        """
        time, event = inverse_transform_survival_target(y_true)
        time = to_tensor(time, device=self.device)
        event = to_tensor(event, device=self.device)
        y_true = torch.stack([time, event], axis=1)
        return self.criterion_(y_pred, y_true)

    def fit(self, X, y=None, **fit_params):
        if not self.warm_start or not self.initialized_:
            self.initialize()

        time, event = inverse_transform_survival_target(y)
        self.train_time = time
        self.train_event = event
        self.partial_fit(X, y, **fit_params)
        self.fit_breslow(
            self.module_.forward(torch.tensor(X)).detach().numpy().ravel(),
            time,
            event,
        )
        return self
