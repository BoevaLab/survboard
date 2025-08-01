library(biomaRt)
library(dplyr)
library(readr)

set.seed(42)

tcga_lihc <- vroom::vroom("data_reproduced/TCGA/LIHC_data_complete_modalities_preprocessed.csv")
icgc_liri <- vroom::vroom("data_reproduced/ICGC/LIRI-JP_data_complete_modalities_preprocessed.csv")


tcga_lihc_clinical <- tcga_lihc[, sapply(strsplit(colnames(tcga_lihc), "_"), function(x) x[[1]]) %in% c("OS", "clinical")][, c(1, 2, 3, 4, 6)]
icgc_liri_clinical <- icgc_liri[, sapply(strsplit(colnames(icgc_liri), "_"), function(x) x[[1]]) %in% c("OS", "clinical")][, c(1, 2, 3, 4, 5)]

colnames(tcga_lihc_clinical)[c(3, 4, 5)] <- c(
  "clinical_age",
  "clinical_gender",
  "clinical_stage"
)


colnames(icgc_liri_clinical)[c(3, 4, 5)] <- c(
  "clinical_age",
  "clinical_gender",
  "clinical_stage"
)


tcga_lihc_clinical$clinical_stage <- ifelse(
  tcga_lihc_clinical$clinical_stage %in% c(".MISSING", "Stage I", "Stage II", "Stage III", "Stage IV"),
  tcga_lihc_clinical$clinical_stage,
  ifelse(
    tcga_lihc_clinical$clinical_stage == "Stage IIIA",
    "Stage III",
    ifelse(
      tcga_lihc_clinical$clinical_stage == "Stage IIIB",
      "Stage III",
      ifelse(
        tcga_lihc_clinical$clinical_stage == "Stage IIIC",
        "Stage III",
        "Stage IV"
      )
    )
  )
)

icgc_liri_clinical$clinical_stage <- ifelse(
  icgc_liri_clinical$clinical_stage == "stage1",
  "Stage I",
  ifelse(
    icgc_liri_clinical$clinical_stage == "stage2",
    "Stage II",
    ifelse(
      icgc_liri_clinical$clinical_stage == "stage3",
      "Stage III",
      "Stage IV"
    )
  )
)

icgc_liri_clinical$clinical_gender <- ifelse(
  icgc_liri_clinical$clinical_gender == "male",
  "MALE",
  "FEMALE"
)

tcga_lihc_gex <- tcga_lihc[, sapply(strsplit(colnames(tcga_lihc), "_"), function(x) x[[1]]) %in% c("gex")]
icgc_liri_gex <- icgc_liri[, sapply(strsplit(colnames(icgc_liri), "_"), function(x) x[[1]]) %in% c("gex")]

tcga_lihc_gex <- tcga_lihc_gex[, which(sapply(strsplit(sapply(strsplit(colnames(tcga_lihc_gex), "_"), function(x) x[[2]]), "\\|"), function(x) x[[1]]) != "?")]
colnames(tcga_lihc_gex) <- sapply(strsplit(colnames(tcga_lihc_gex), "\\|"), function(x) x[[1]])

intersected_colnames <- intersect(colnames(tcga_lihc_gex), colnames(icgc_liri_gex))[-15545]

tcga_lihc_gex <- tcga_lihc_gex[, sapply(intersected_colnames, function(x) which(x == colnames(tcga_lihc_gex)))]

icgc_liri_gex <- icgc_liri_gex[, sapply(intersected_colnames, function(x) which(x == colnames(icgc_liri_gex)))]

qn_matrix <- cbind(t(icgc_liri_gex), t(tcga_lihc_gex))

# From: https://bioinformatics.stackexchange.com/questions/6863/how-to-quantile-normalization-on-rna-seq-counts
quantile_normalisation <- function(df) {
  df_rank <- apply(df, 2, rank, ties.method = "min")
  df_sorted <- data.frame(apply(df, 2, sort))
  df_mean <- apply(df_sorted, 1, mean)

  index_to_mean <- function(my_index, my_mean) {
    return(my_mean[my_index])
  }

  df_final <- apply(df_rank, 2, index_to_mean, my_mean = df_mean)
  rownames(df_final) <- rownames(df)
  return(df_final)
}

quantile_normalized <- data.frame(t(quantile_normalisation(data.frame(qn_matrix))))

gex_liri_qn <- quantile_normalized[1:nrow(icgc_liri_gex), ]
gex_lihc_qn <- quantile_normalized[-(1:nrow(icgc_liri_gex)), ]

liri_finalized <- cbind(icgc_liri_clinical, gex_liri_qn)

lihc_finalized <- cbind(tcga_lihc_clinical, gex_lihc_qn)

liri_finalized %>% write_csv(
  "data_reproduced/validation/LIHC/icgc_liri_jp.csv"
)

lihc_finalized %>% write_csv(
  "data_reproduced/validation/LIHC/tcga_lihc.csv"
)


tcga_paad <- vroom::vroom("data_reproduced/TCGA/PAAD_data_complete_modalities_preprocessed.csv")
icgc_paad_ca <- vroom::vroom("data_reproduced/ICGC/PACA-CA_data_complete_modalities_preprocessed.csv")

tcga_paad_clinical <- tcga_paad[, sapply(strsplit(colnames(tcga_paad), "_"), function(x) x[[1]]) %in% c("OS", "clinical")][, c(1:4, 6)]
icgc_ca_clinical <- icgc_paad_ca[, sapply(strsplit(colnames(icgc_paad_ca), "_"), function(x) x[[1]]) %in% c("OS", "clinical")][, 1:5]

colnames(tcga_paad_clinical)[c(3:5)] <- c(
  "clinical_age",
  "clinical_gender",
  "clinical_tumor_stage"
)

colnames(icgc_ca_clinical)[c(3, 4, 5)] <- c(
  "clinical_age",
  "clinical_gender",
  "clinical_tumor_stage"
)

icgc_ca_clinical$clinical_gender <- ifelse(
  icgc_ca_clinical$clinical_gender == "male",
  "MALE",
  "FEMALE"
)

icgc_ca_clinical$clinical_tumor_stage <- ifelse(
  icgc_ca_clinical$clinical_tumor_stage %in% paste0("stage", c("IA", "IB", "II", "IIA", "IIB", "III")),
  paste0("Stage ", sapply(strsplit(icgc_ca_clinical$clinical_tumor_stage, "stage"), function(x) x[[2]])),
  ".MISSING"
)


icgc_ca_gex <- icgc_paad_ca[, sapply(strsplit(colnames(icgc_paad_ca), "_"), function(x) x[[1]]) %in% c("gex")]
tcga_paad_gex <- tcga_paad[, sapply(strsplit(colnames(tcga_paad), "_"), function(x) x[[1]]) %in% c("gex")]


tcga_paad_gex <- tcga_paad_gex[, which(sapply(strsplit(sapply(strsplit(colnames(tcga_paad_gex), "_"), function(x) x[[2]]), "\\|"), function(x) x[[1]]) != "?")]
colnames(tcga_paad_gex) <- sapply(strsplit(colnames(tcga_paad_gex), "\\|"), function(x) x[[1]])

mart <- useDataset("hsapiens_gene_ensembl", useMart("ensembl"))
genes <- sapply(strsplit(colnames(icgc_ca_gex), "_"), function(x) x[[2]])
G_list <- getBM(filters = "ensembl_gene_id", attributes = c("ensembl_gene_id", "hgnc_symbol"), values = genes, mart = mart)

colnames(icgc_ca_gex) <- paste0("gex_", G_list$hgnc_symbol)

intersected_colnames <- intersect(colnames(icgc_ca_gex), colnames(tcga_paad_gex))

intersected_colnames <- intersected_colnames[!intersected_colnames %in% c(
  "gex_SIGLEC5", "gex_SNTB2", "gex_SNORD38B", "gex_PINX1", "gex_SNORA16A", "gex_SCARNA4"
)]

tcga_paad_gex <- tcga_paad_gex[, sapply(intersected_colnames, function(x) which(x == colnames(tcga_paad_gex)))]

icgc_ca_gex <- icgc_ca_gex[, sapply(intersected_colnames, function(x) which(x == colnames(icgc_ca_gex)))]

qn_matrix <- cbind(t(icgc_ca_gex), t(tcga_paad_gex))

# From: https://bioinformatics.stackexchange.com/questions/6863/how-to-quantile-normalization-on-rna-seq-counts
quantile_normalisation <- function(df) {
  df_rank <- apply(df, 2, rank, ties.method = "min")
  df_sorted <- data.frame(apply(df, 2, sort))
  df_mean <- apply(df_sorted, 1, mean)

  index_to_mean <- function(my_index, my_mean) {
    return(my_mean[my_index])
  }

  df_final <- apply(df_rank, 2, index_to_mean, my_mean = df_mean)
  rownames(df_final) <- rownames(df)
  return(df_final)
}

quantile_normalized <- data.frame(t(quantile_normalisation(data.frame(qn_matrix))))

gex_ca_qn <- quantile_normalized[1:nrow(icgc_ca_gex), ]
gex_paad_qn <- quantile_normalized[-(1:nrow(icgc_ca_gex)), ]

ca_finalized <- cbind(icgc_ca_clinical, gex_ca_qn)

paad_finalized <- cbind(tcga_paad_clinical, gex_paad_qn)

ca_finalized %>% write_csv(
  "data_reproduced/validation/PAAD/icgc_paca_ca.csv"
)

paad_finalized %>% write_csv(
  "data_reproduced/validation/PAAD/tcga_paad.csv"
)

sessionInfo()
