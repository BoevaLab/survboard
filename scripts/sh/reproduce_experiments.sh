#!/usr/bin/env bash

Rscript ./scripts/R/benchmark_unimodal.R
Rscript ./scripts/R/benchmark_pancan.R
Rscript ./scripts/R/benchmark_multimodal_missing.R
Rscript ./scripts/R/benchmark_multimodal.R

python ./scripts/driver_unimodal.py
python ./scripts/driver_pancan.py
python ./scripts/driver_multimodal_missing.py
python ./scripts/driver_multimodal_clinical_gex.py
python ./scripts/driver_multimodal_all.py
