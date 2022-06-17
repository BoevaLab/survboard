import numpy as np
from pycox.evaluation.eval_surv import EvalSurv
from survival_evaluation.evaluations import d_calibration


class EvalSurvDCalib(EvalSurv):
    def d_calibration_(self, bins=10, p_value=False):
        indices = self.idx_at_times(self.durations)

        d_calib = d_calibration(
            self.events,
            np.array(
                [
                    self.surv.iloc[indices[ix], ix]
                    for ix in range(self.events.shape[0])
                ]
            ),
            bins=bins,
        )
        if p_value:
            d_calib["p_value"]
        else:
            return d_calib["test_statistic"]
