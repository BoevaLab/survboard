#!/usr/bin/env bash

python ./scripts/python/create_master.py
python ./scripts/python/evaluate_metrics.py

python ./scripts/python/reproduce_table_S1.py

Rscript ./scripts/R/plot_figure_2.R
Rscript ./scripts/R/plot_figure_4.R
Rscript ./scripts/R/plot_figure_5.R

Rscript ./scripts/R/plot_figure_S7.R
Rscript ./scripts/R/plot_figure_S8.R
Rscript ./scripts/R/plot_figure_S9.R
Rscript ./scripts/R/plot_figure_S10.R

Rscript ./scripts/R/make_table_S3.R
Rscript ./scripts/R/make_table_S4.R
