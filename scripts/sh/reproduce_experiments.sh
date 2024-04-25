#!/usr/bin/env bash

Rscript ./scripts/R/benchmark_unimodal.R
Rscript ./scripts/R/benchmark_pancan.R
Rscript ./scripts/R/benchmark_multimodal_missing.R
Rscript ./scripts/R/benchmark_multimodal.R

python ./scripts/python/driver_multimodal_all.py
python ./scripts/python/driver_multimodal_clinical_gex.py

for modalities in "clinical" "mirna" "rppa" "gex" "mut" "meth" "cnv"; do
    python ./scripts/python/driver_unimodal.py --modalities $modalities
done

for model in "cox" "eh"; do
    for fusion in "intermediate_concat" "late_mean"; do
        python ./scripts/python/driver_pancan.py --model_type $model --fusion $fusion
    done
done

for fusion in "intermediate_concat" "late_mean"; do
    python ./scripts/python/driver_multimodal_missing.py --fusion $fusion --project "METABRIC" --cancer "BRCA"

    for cancer in "LIRI-JP" "PACA-CA" "PACA-AU" "CLLE-ES"; do
        python ./scripts/python/driver_multimodal_missing.py --fusion $fusion --project "ICGC" --cancer $cancer

    done

    for cancer in "ALL" "WT"; do
        python ./scripts/python/driver_multimodal_missing.py --fusion $fusion --project "TARGET" --cancer $cancer
    done

    for cancer in "BLCA" "BRCA" "COAD" "ESCA" "HNSC" "KIRC" "KIRP" "LGG" "LUAD" "PAAD" \
        "SARC" "SKCM" "STAD" "UCEC" "OV" "LIHC" "LUSC" "LAML" "CESC" "GBM" "READ"; do
        python ./scripts/python/driver_multimodal_missing.py --fusion $fusion --project "TCGA" --cancer $cancer
    done
done
