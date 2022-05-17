import sys

import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from survival_benchmark.survival_benchmark.python.modules import (
    cheerla_et_al,
    CheerlaEtAlNet,
    CoxPHNet,
    DeepSurv,
    GDPNet,
    GDP
)
from survival_benchmark.survival_benchmark.python.utils import (
    cheerla_et_al_criterion,
    get_blocks,
    cox_criterion,
    run_benchmark
)

from hyperband import HyperbandSearchCV

def main():
    param_distributions = {}
    # Read in data
    models = [
        make_pipeline(
            StandardScaler(),
            CheerlaEtAlNet(
                module=cheerla_et_al,
                module__blocks=[],  # TODO: Set during each loop
                module__encoding_dimension=128,
                module__p_multimodal_dropout=0.25,
                criterion=cheerla_et_al_criterion,
                lr=0.01,
                max_epochs=1,
            ),
        ),
        make_pipeline(
            StandardScaler(),
            CoxPHNet(
                module=DeepSurv,
                module__input_dimension=0,  # TODO: Set during each loop
                module__hidden_layer_sizes=[256, 128],
                criterion=cox_criterion,
                lr=0.01,
                max_epochs=1,
            ),
        ),
        make_pipeline(
            StandardScaler(),
            GDPNet(
                module=GDP,
                module__hidden_layer_sizes=[256, 128],
                module__blocks=get_blocks(data.columns),
                criterion=cox_criterion,
                lr=0.001,
                max_epochs=3,
            ),
        ),
    ]
    for ix, model in enumerate(models):
        for cancer in config["icgc_cancers"]:
            data = pd.read_csv(
                "~/boeva_lab_scratch/data/projects/David/Nikita_David_survival_benchmark/survival_benchmark/data/processed/TARGET/ALL_data_complete_modalities_preprocessed.csv"
            ).iloc[:, 1:]
            time, event = data["OS_days"].values, data["OS"].values
            data = data.iloc[:, 3:]
            data = pd.get_dummies(data.fillna("NA"))
            grid = HyperbandSearchCV(
                estimator=model,
                param_distributions=param_distributions[ix],
                resource_params="module__max_epochs",
                scoring="TODO",
                iid=False,
                cv="TODO",
                random_state=config["seed"],
            )
            est = grid.best_estimator_

            # TODO: Add batchnorm to everything
            # TODO: Clean up code for modules
            # TODO: Clean up driver code
            # TODO: Write driver code for pancancer
            # TODO: Write driver code for missing modalities


if __name__ == "__main__":
    sys.exit(main())
