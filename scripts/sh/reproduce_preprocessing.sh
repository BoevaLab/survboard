#!/usr/bin/env bash

Rscript ./scripts/R/run_preprocessing_R.R \
    --keep_non_primary_samples false \
    --keep_patients_without_survival_information false

python ./scripts/python/rerun_splits.py
Rscript ./scripts/R/create_pancancer_data.R
Rscript ./scripts/R/extract_pancancer_master.R
Rscript ./scripts/R/extract_pancancer_splits.R
Rscript ./scripts/R/extract_master.R
