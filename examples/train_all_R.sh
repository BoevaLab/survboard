#!/bin/sh

for cancer in BLCA BRCA COAD ESCA HNSC KIRC KIRP LGG LUAD PAAD \
    SARC SKCM STAD UCEC OV LIHC LUSC LAML CESC GBM READ; do
    Rscript scripts/R_standard_example_driver.R \
        --project TCGA
        --cancer ${cancer}
        --n_cores 8
done

for cancer in CLLE-ES PACA-AU PACA-CA LIRI-JP; do
    Rscript scripts/R_standard_example_driver.R \
        --project ICGC
        --cancer ${cancer}
        --n_cores 8
done

for cancer in ALL AML WT; do
    Rscript scripts/R_standard_example_driver.R \
        --project TARGET
        --cancer ${cancer}
        --n_cores 8
done
