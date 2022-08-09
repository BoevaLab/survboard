#!/usr/bin/env bash

Rscript ../R/run_preprocessing_R.R \
    --keep_non_primary_samples false \
    --keep_patients_without_survival_information false

python ../python/target_preprocessor.py \
../../data_template/TARGET
