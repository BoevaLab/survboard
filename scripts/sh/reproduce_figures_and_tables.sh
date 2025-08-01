#!/usr/bin/env bash

python ./scripts/python/evaluate_metrics.py

Rscript ./scripts/R/plot_figure_3.R
Rscript ./scripts/R/plot_figure_4.R
Rscript ./scripts/R/plot_figure_5.R

Rscript ./scripts/R/plot_figure_S1.R
Rscript ./scripts/R/plot_figure_S2.R
Rscript ./scripts/R/plot_figure_S3.R
Rscript ./scripts/R/plot_figure_S4.R
Rscript ./scripts/R/plot_figure_S5.R

python ./scripts/python/reproduce_table_S2.py
Rscript ./scripts/R/reproduce_table_S3.R
