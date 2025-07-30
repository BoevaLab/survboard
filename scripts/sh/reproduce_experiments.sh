#!/usr/bin/env bash

Rscript ./scripts/R/benchmark_unimodal.R
Rscript ./scripts/R/benchmark_pancan.R
Rscript ./scripts/R/benchmark_multimodal_missing.R
Rscript ./scripts/R/benchmark_multimodal.R
Rscript ./scripts/R/benchmark_multimodal_clinical_gex_transfer.R

for modalities in "clinical" "mirna" "rppa" "gex" "mut" "meth" "cnv"; do
    python ./scripts/python/driver_unimodal.py --modalities $modalities
done

python ./scripts/python/driver_multimodal_all.py
python ./scripts/python/driver_multimodal_clinical_gex.py

for split in {1..25}; do
  python ./scripts/python/driver_unimodal.py --project METABRIC --cancer BRCA --split $split
  python ./scripts/python/driver_multimodal_missing_revisions.py --project METABRIC --cancer $cancer --split $split
  python ./scripts/python/driver_clinical_gex_revisions.py --project METABRIC --cancer $cancer --split $split
  python ./scripts/python/driver_multimodal_all_revisions.py --project METABRIC --cancer $cancer --split $split
done

Rscript make_salmon_data_non_split.R METABRIC $cancer
python ./scripts/python/driver_clinical_gex_salmon.py --project METABRIC --cancer $cancer
python ./scripts/python/driver_multimodal_salmon.py --project METABRIC --cancer $cancer
Rscript scripts/R/run_denoising_ae_full.R METABRIC $cancer
Rscript scripts/R/run_denoising_ae_clinical_gex.R METABRIC $cancer
python ./scripts/python/driver_multimodal_customics.py --project METABRIC --cancer $cancer
python ./scripts/python/driver_clinical_gex_customics.py --project METABRIC --cancer $cancer
python ./scripts/python/driver_clinical_gex_multimodal_survival_pred.py --project $METABRIC --cancer $cancer
python ./scripts/python/driver_multimodal_multimodal_survival_pred.py --project $METABRIC --cancer $cancer

python ./scripts/python/driver_clinical_gex_gdp.py --project METABRIC --cancer $cancer
python ./scripts/python/driver_multimodal_gdp_fixed.py --project METABRIC --cancer $cancer
python ./scripts/python/driver_clinical_gex_survival_net.py --project METABRIC --cancer $cancer
python ./scripts/python/driver_multimodal_survival_net.py --project METABRIC --cancer $cancer

for cancer in "CLLE-ES" "PACA-AU" "PACA-CA" "LIRI-JP"; do
    for split in {1..25}; do
        python ./scripts/python/driver_unimodal.py --project ICGC --cancer $cancer --split $split
        python ./scripts/python/driver_multimodal_missing_revisions.py --project ICGC --cancer $cancer --split $split
        python ./scripts/python/driver_clinical_gex_revisions.py --project ICGC --cancer $cancer --split $split
        python ./scripts/python/driver_multimodal_all_revisions.py --project ICGC --cancer $cancer --split $split
    done
    Rscript make_salmon_data_non_split.R ICGC $cancer
    python ./scripts/python/driver_clinical_gex_salmon.py --project ICGC --cancer $cancer
    python ./scripts/python/driver_multimodal_salmon.py --project ICGC --cancer $cancer
    Rscript scripts/R/run_denoising_ae_full.R ICGC $cancer
    Rscript scripts/R/run_denoising_ae_clinical_gex.R ICGC $cancer
    python ./scripts/python/driver_multimodal_customics.py --project ICGC --cancer $cancer
    python ./scripts/python/driver_clinical_gex_customics.py --project ICGC --cancer $cancer
    python ./scripts/python/driver_clinical_gex_multimodal_survival_pred.py --project $ICGC --cancer $cancer
    python ./scripts/python/driver_multimodal_multimodal_survival_pred.py --project $ICGC --cancer $cancer

    python ./scripts/python/driver_clinical_gex_gdp.py --project ICGC --cancer $cancer
    python ./scripts/python/driver_multimodal_gdp_fixed.py --project ICGC --cancer $cancer
    python ./scripts/python/driver_clinical_gex_survival_net.py --project ICGC --cancer $cancer
    python ./scripts/python/driver_multimodal_survival_net.py --project ICGC --cancer $cancer
done

for cancer in "WT" "ALL"; do
    for split in {1..25}; do
        python ./scripts/python/driver_unimodal.py --project TARGET --cancer $cancer --split $split
        python ./scripts/python/driver_multimodal_missing_revisions.py --project TARGET --cancer $cancer --split $split
        python ./scripts/python/driver_clinical_gex_revisions.py --project TARGET --cancer $cancer --split $split
        python ./scripts/python/driver_multimodal_all_revisions.py --project TARGET --cancer $cancer --split $split
    done
    Rscript make_salmon_data_non_split.R TARGET $cancer
    python ./scripts/python/driver_clinical_gex_salmon.py --project TARGET --cancer $cancer
    python ./scripts/python/driver_multimodal_salmon.py --project TARGET --cancer $cancer
    Rscript scripts/R/run_denoising_ae_full.R TARGET $cancer
    Rscript scripts/R/run_denoising_ae_clinical_gex.R TARGET $cancer
    python ./scripts/python/driver_multimodal_customics.py --project TARGET --cancer $cancer
    python ./scripts/python/driver_clinical_gex_customics.py --project TARGET --cancer $cancer
    python ./scripts/python/driver_clinical_gex_multimodal_survival_pred.py --project $TARGET --cancer $cancer
    python ./scripts/python/driver_multimodal_multimodal_survival_pred.py --project $TARGET --cancer $cancer

    python ./scripts/python/driver_clinical_gex_gdp.py --project TARGET --cancer $cancer
    python ./scripts/python/driver_multimodal_gdp_fixed.py --project TARGET --cancer $cancer
    python ./scripts/python/driver_clinical_gex_survival_net.py --project TARGET --cancer $cancer
    python ./scripts/python/driver_multimodal_survival_net.py --project TARGET --cancer $cancer
done

for cancer in "BLCA" "BRCA" "COAD" "ESCA" "HNSC" "KIRC" "KIRP" "LGG" "LUAD" "PAAD" "SARC" "SKCM" \
        "STAD" "UCEC" "OV" "LIHC" "LUSC" "LAML" "CESC" "GBM" "READ"; do
    for split in {1..25}; do
        python ./scripts/python/driver_unimodal.py --project TCGA --cancer $cancer --split $split
        python ./scripts/python/driver_multimodal_missing_revisions.py --project TCGA --cancer $cancer --split $split
        python ./scripts/python/driver_clinical_gex_revisions.py --project TCGA --cancer $cancer --split $split
        python ./scripts/python/driver_multimodal_all_revisions.py --project TCGA --cancer $cancer --split $split
    done
    Rscript make_salmon_data_non_split.R TCGA $cancer
    python ./scripts/python/driver_clinical_gex_salmon.py --project TCGA --cancer $cancer
    python ./scripts/python/driver_multimodal_salmon.py --project TCGA --cancer $cancer
    Rscript scripts/R/run_denoising_ae_full.R TCGA $cancer
    Rscript scripts/R/run_denoising_ae_clinical_gex.R TCGA $cancer
    python ./scripts/python/driver_multimodal_customics.py --project TCGA --cancer $cancer
    python ./scripts/python/driver_clinical_gex_customics.py --project TCGA --cancer $cancer
    python ./scripts/python/driver_clinical_gex_multimodal_survival_pred.py --project $TCGA --cancer $cancer
    python ./scripts/python/driver_multimodal_multimodal_survival_pred.py --project $TCGA --cancer $cancer

    python ./scripts/python/driver_clinical_gex_gdp.py --project TCGA --cancer $cancer
    python ./scripts/python/driver_multimodal_gdp_fixed.py --project TCGA --cancer $cancer
    python ./scripts/python/driver_clinical_gex_survival_net.py --project TCGA --cancer $cancer
    python ./scripts/python/driver_multimodal_survival_net.py --project TCGA --cancer $cancer
done

for split in {1..25}; do
    python ./scripts/python/driver_pancan.py --split $split
done

snakemake --rerun-incomplete
