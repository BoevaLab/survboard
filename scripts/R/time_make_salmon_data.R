log <- file(snakemake@log[[1]], open = "wt")
sink(log, type = "output")
sink(log, type = "message")

.libPaths(c("/cluster/customapps/biomed/boeva/dwissel/4.2", .libPaths()))
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

options <- commandArgs(trailingOnly = TRUE)

for (project in c("TCGA")) {
  for (cancer in c(snakemake@wildcards[["cancer"]])) {
    set.seed(42)
    # Read in complete modality sample dataset.
    data <- vroom::vroom(
      here::here(
        "data_reproduced", project,
        paste0(cancer, "_data_complete_modalities_preprocessed.csv", collapse = "")
      )
    )
    # Remove patient_id column and explicitly cast character columns as strings.
    data <- data.frame(data[, -which("patient_id" == colnames(data))]) %>%
      mutate(across(where(is.character), as.factor))

    modalities <- sapply(strsplit(colnames(data), "\\_"), function(x) x[[1]])
    data <- data[, modalities %in% c("OS", "clinical", "gex")]

    label <- data[, c(which("OS" == colnames(data)), which("OS_days" == colnames(data)))]
    data <- data[, -c(which("OS" == colnames(data)), which("OS_days" == colnames(data)))]

    transformed_data <- transform_matrix_salmon(data)
    finalized <- cbind(label, transformed_data)

    write.table(data.frame(), file = here::here(
      "results_reproduced", "timings", paste0("make_salmon_data_", snakemake@wildcards[["cancer"]])
    ), col.names = FALSE)
  }
}

sessionInfo()

sink()
sink()
