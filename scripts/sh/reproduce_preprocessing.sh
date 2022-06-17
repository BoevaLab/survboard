#!/usr/bin/env bash

Rscript ../R/recreate_tcga_icgc.R

python ../python/target_preprocessor.py \
../../data_template/TARGET
