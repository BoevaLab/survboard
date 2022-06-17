#!/usr/bin/env bash

Rscript ../R/benchmark_all.R
Rscript ../R/benchmark_clinical_all.R
Rscript ../R/benchmark_modality_experiments.R

python ../python/scripts/driver_all.py \
    ../../data \
    ../../config/config.json ../../config/params.json \
    ../../results_reproduced \
    naive TCGA standard impute

python ../python/scripts/driver_all.py \
    ../../data \
    ../../config/config.json ../../config/params.json \
    ../../results_reproduced \
    mean TCGA standard impute

python ../python/scripts/driver_all.py \
    ../../data \
    ../../config/config.json ../../config/params.json \
    ../../results_reproduced \
    mean TCGA missing multimodal_dropout

python ../python/scripts/driver_all.py \
    ../../data \
    ../../config/config.json ../../config/params.json \
    ../../results_reproduced \
    mean TCGA pancancer multimodal_dropout

python ../python/scripts/driver_all.py \
    ../../data \
    ../../config/config.json ../../config/params.json \
    ../../results_reproduced \
    naive ICGC standard impute

python ../python/scripts/driver_all.py \
    ../../data \
    ../../config/config.json ../../config/params.json \
    ../../results_reproduced \
    mean ICGC standard impute

python ../python/scripts/driver_all.py \
    ../../data \
    ../../config/config.json ../../config/params.json \
    ../../results_reproduced \
    naive TARGET standard impute

python ../python/scripts/driver_all.py \
    ../../data \
    ../../config/config.json ../../config/params.json \
    ../../results_reproduced \
    mean TARGET standard impute

python ../python/scripts/driver_all.py \
    ../../data \
    ../../config/config.json ../../config/params.json \
    ../../results_reproduced \
    naive TARGET standard impute

python ../python/scripts/driver_gex.py \
    ../../data \
    ../../config/config.json ../../config/params.json \
    ../../results_reproduced \
    naive TCGA standard impute

python ../python/scripts/driver_gex_clinical.py \
    ../../data \
    ../../config/config.json ../../config/params.json \
    ../../results_reproduced \
    naive TCGA standard impute

python ../python/scripts/driver_gex_clinical.py \
    ../../data \
    ../../config/config.json ../../config/params.json \
    ../../results_reproduced \
    mean TCGA standard impute

python ../python/scripts/driver_multi_omics_without_clinical.py.py \
    ../../data \
    ../../config/config.json ../../config/params.json \
    ../../results_reproduced \
    naive TCGA standard impute

python ../python/scripts/driver_multi_omics_without_clinical.py.py \
    ../../data \
    ../../config/config.json ../../config/params.json \
    ../../results_reproduced \
    mean TCGA standard impute

python ../python/scripts/driver_all_alpha_ablation.py.py \
    ../../data \
    ../../config/config.json ../../config/params.json \
    ../../results_reproduced/alpha_ablation \
    mean TCGA standard impute

python ../python/scripts/driver_all_alpha_ablation.py.py \
    ../../data \
    ../../config/config.json ../../config/params.json \
    ../../results_reproduced/alpha_ablation \
    mean ICGC standard impute

python ../python/scripts/driver_all_alpha_ablation.py.py \
    ../../data \
    ../../config/config.json ../../config/params.json \
    ../../results_reproduced/alpha_ablation \
    mean TARGET standard impute