#!/bin/bash

Rscript ./scripts/R/run_preprocessing_R.R \
    --keep_non_primary_samples false \
    --keep_patients_without_survival_information false

python scripts/python/rerun_splits.py
Rscript ./scripts/R/driver_multimodal_missing.py
python scripts/python/create_master.py
