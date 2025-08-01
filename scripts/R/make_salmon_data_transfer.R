suppressPackageStartupMessages({
  library(lmQCM)
  library(vroom)
  library(rjson)
  library(dplyr)
  library(readr)
  source(here::here("survboard", "R", "utils", "utils.R"))
})

config <- rjson::fromJSON(
  file = here::here("config", "config.json")
)

# Seeding for reproducibility.
set.seed(42)


transform_matrix_salmon <- function(data) {
  modalities <- unique(sapply(strsplit(colnames(data), "\\_"), function(x) x[[1]]))

  for (modality in c("gex", "clinical", "mirna", "meth", "rppa", "mut", "cnv")) {
    if (modality == "gex" & "gex" %in% modalities) {
      mod <- data[, which(sapply(strsplit(colnames(data), "\\_"), function(x) x[[1]]) == modality)]
      mod_ready <- t(mod)

      mod_ready_filtered <- fastFilter(mod_ready, 0.2, 0.2)


      mod_finished <- lmQCM(
        mod_ready_filtered,
        gamma = 0.7,
        lambda = 1,
        t = 1,
        beta = 0.4,
        minClusterSize = 10,
        CCmethod = "spearman"
      )
      mod_matrix <- data.frame(t(mod_finished@eigengene.matrix))
      colnames(mod_matrix) <- paste0("gex_", 1:ncol(mod_matrix))
      transformed_matrix <- mod_matrix
    }
    if (modality == "clinical" & "clinical" %in% modalities) {
      transformed_matrix <- cbind(transformed_matrix, data[, which(sapply(strsplit(colnames(data), "\\_"), function(x) x[[1]]) == "clinical")])
    }
    if (modality == "mirna" & "mirna" %in% modalities) {
      mod <- data[, which(sapply(strsplit(colnames(data), "\\_"), function(x) x[[1]]) == modality)]
      mod_ready <- t(mod)

      mod_ready_filtered <- fastFilter(mod_ready, 0.2, 0.2)


      mod_finished <- lmQCM(
        mod_ready_filtered,
        gamma = 0.4,
        lambda = 1,
        t = 1,
        beta = 0.6,
        minClusterSize = 4,
        CCmethod = "spearman"
      )
      mod_matrix <- data.frame(t(mod_finished@eigengene.matrix))
      colnames(mod_matrix) <- paste0("mirna_", 1:ncol(mod_matrix))
      transformed_matrix <- cbind(transformed_matrix, mod_matrix)
    }
    if (modality == "meth" & "meth" %in% modalities) {
      mod <- data[, which(sapply(strsplit(colnames(data), "\\_"), function(x) x[[1]]) == modality)]
      mod_ready <- t(mod)

      mod_ready_filtered <- fastFilter(mod_ready, 0.2, 0.2)

      mod_finished <- lmQCM(
        mod_ready_filtered,
        gamma = 0.7,
        lambda = 1,
        t = 1,
        beta = 0.4,
        minClusterSize = 10,
        CCmethod = "spearman"
      )
      mod_matrix <- data.frame(t(mod_finished@eigengene.matrix))
      colnames(mod_matrix) <- paste0("meth_", 1:ncol(mod_matrix))
      transformed_matrix <- cbind(transformed_matrix, mod_matrix)
    }
    if (modality == "rppa" & "rppa" %in% modalities) {
      mod <- data[, which(sapply(strsplit(colnames(data), "\\_"), function(x) x[[1]]) == modality)]
      mod_ready <- t(mod)

      mod_ready_filtered <- fastFilter(mod_ready, 0.2, 0.2)


      mod_finished <- lmQCM(
        mod_ready_filtered,
        gamma = 0.4,
        lambda = 1,
        t = 1,
        beta = 0.6,
        minClusterSize = 4,
        CCmethod = "spearman"
      )
      mod_matrix <- data.frame(t(mod_finished@eigengene.matrix))
      colnames(mod_matrix) <- paste0("rppa_", 1:ncol(mod_matrix))
      transformed_matrix <- cbind(transformed_matrix, mod_matrix)
    }
    if (modality == "mut" & "mut" %in% modalities) {
      transformed_matrix <- cbind(transformed_matrix, data.frame(mut_1 = apply(data[, which(sapply(strsplit(colnames(data), "\\_"), function(x) x[[1]]) == "mut")], 1, function(x) sum(abs(x)))))
    }
    if (modality == "cnv" & "cnv" %in% modalities) {
      transformed_matrix <- cbind(transformed_matrix, data.frame(cnv_1 = apply(data[, which(sapply(strsplit(colnames(data), "\\_"), function(x) x[[1]]) == "cnv")], 1, function(x) sum(abs(x)))))
    }
  }
  return(transformed_matrix)
}



for (project in c("validation")) {
  for (cancer in c("PAAD", "LIHC")) {
    target_dir <- paste0("results_reproduced/survival_functions/transfer/", "validation", "/", cancer, "/", "multimodal_nsclc")
    if (!dir.exists(target_dir)) {
      print(target_dir)
      dir.create(target_dir)
    }
    if (cancer == "PAAD") {
      # Read in complete modality sample dataset.
      data_input <- vroom::vroom(
        here::here(
          "data_reproduced", project,
          cancer,
          "icgc_paca_ca.csv"
        )
      )

      data_transfer_to <- vroom::vroom(
        here::here(
          "data_reproduced", project,
          cancer,
          "tcga_paad.csv"
        )
      )
      data_transfer_to$clinical_tumor_stage <- as.factor(data_transfer_to$clinical_tumor_stage)
      data_transfer_to$clinical_gender <- as.factor(data_transfer_to$clinical_gender)

      data_input$clinical_tumor_stage <- as.factor(data_input$clinical_tumor_stage)
      data_input$clinical_gender <- as.factor(data_input$clinical_gender)

      split_project <- "ICGC"
      split_cancer <- "PACA-CA"
    } else {
      data_input <- vroom::vroom(
        here::here(
          "data_reproduced", project,
          cancer,
          "tcga_lihc.csv"
        )
      )

      data_transfer_to <- vroom::vroom(
        here::here(
          "data_reproduced", project,
          cancer,
          "icgc_liri_jp.csv"
        )
      )
      data_transfer_to$clinical_stage <- as.factor(data_transfer_to$clinical_stage)
      data_transfer_to$clinical_gender <- as.factor(data_transfer_to$clinical_gender)

      data_input$clinical_stage <- as.factor(data_input$clinical_stage)
      data_input$clinical_gender <- as.factor(data_input$clinical_gender)
      split_project <- "TCGA"
      split_cancer <- "LIHC"
    }

    data <- rbind(data_input, data_transfer_to)
    label <- data[, c(which("OS" == colnames(data)), which("OS_days" == colnames(data)))]
    data <- data[, -c(which("OS" == colnames(data)), which("OS_days" == colnames(data)))]

    transformed_data <- transform_matrix_salmon(data)
    finalized <- cbind(label, transformed_data)

    finalized %>% write_tsv(
      here::here(
        "data_reproduced", project,
        paste0(cancer, "_salmon_preprocessed.csv", collapse = "")
      )
    )
  }
}
