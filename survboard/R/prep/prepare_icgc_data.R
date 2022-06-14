library(rjson)
library(vroom)
library(here)
library(dplyr)
library(readr)
library(tibble)
library(tidyr)
library(fastDummies)
library(janitor)
library(maftools)
library(reshape2)

impute_icgc <- function(df) {
  big_missing_genes <- which(apply(df, 2, function(x) sum(is.na(x))) > round(dim(df)[1] / 10))
  if (length(big_missing_genes) > 0) {
    df <- df[, -big_missing_genes]
  }

  median_per_gene <- apply(df, 2, function(x) median(x, na.rm = TRUE))
  median_per_gene <- split(unname(median_per_gene), names(median_per_gene))
  df <- df %>% replace_na(median_per_gene)
  return(df)
}

filter_out_duplicates_icgc <- function(df) {
  main_frame <- df
  indicator_frame <- df %>%
    group_by(icgc_donor_id) %>%
    summarise(
      n_samples = n_distinct(icgc_sample_id),
      n_specimens = n_distinct(icgc_specimen_id)
    )

  duplicated_specimen_and_sample_donors <- indicator_frame %>%
    filter(n_samples > n_specimens & n_specimens > 1) %>%
    pull(icgc_donor_id)

  duplicated_specimen_donors <- indicator_frame %>%
    filter(n_specimens > 1) %>%
    pull(icgc_donor_id)

  specimens_to_remove <- main_frame %>%
    filter(icgc_donor_id %in% duplicated_specimen_donors) %>%
    group_by(icgc_donor_id) %>%
    filter(specimen_interval != min(specimen_interval, na.rm = TRUE) | is.na(specimen_interval)) %>%
    pull(icgc_specimen_id)

  df <- df %>% filter(!icgc_specimen_id %in% specimens_to_remove)
  duplicated_sample_donors <- indicator_frame %>%
    filter(n_samples > 1 & n_specimens == 1) %>%
    pull(icgc_donor_id)

  samples_to_remove <- main_frame %>%
    filter(icgc_donor_id %in% duplicated_sample_donors) %>%
    group_by(icgc_donor_id) %>%
    filter(analyzed_sample_interval != min(analyzed_sample_interval, na.rm = TRUE) | is.na(analyzed_sample_interval)) %>%
    pull(icgc_sample_id)

  df <- df %>% filter(!icgc_sample_id %in% samples_to_remove)
  if (length(setdiff(unique(main_frame$icgc_donor_id), unique(df$icgc_donor_id))) > 1) {
    duplicated_sample_frame <- main_frame %>%
      filter(
        icgc_donor_id %in% setdiff(unique(main_frame$icgc_donor_id), unique(df$icgc_donor_id))
      ) %>%
      group_by(icgc_donor_id) %>%
      slice_sample(n = 1) %>%
      ungroup()
    df <- rbind(df, duplicated_sample_frame)
  }
  indicator_frame <- df %>%
    group_by(icgc_donor_id) %>%
    summarise(
      n_samples = n_distinct(icgc_sample_id),
      n_specimens = n_distinct(icgc_specimen_id)
    )
  if (any(indicator_frame$n_samples > 1) | any(indicator_frame$n_specimens > 1)) {
    df <- df %>%
      group_by(icgc_donor_id) %>%
      slice_sample(n = 1) %>%
      ungroup()
  }

  if ("tbl" %in% class(df)) {
    df <- data.frame(df, check.names = FALSE)
  }

  return(df)
}

prepare_mutation_icgc <- function(mut) {
  mut <- filter_out_duplicates_icgc(mut)
  mut <- mut %>% dplyr::select(icgc_donor_id, starts_with("ENSG"))
  rownames(mut) <- mut[, 1]
  mut <- mut[, -1]
  if (any(is.na(mut))) {
    mut <- impute_icgc(mut)
  }
  return(mut)
}

add_missing_modality_samples_icgc <- function(df, clinical_donor_ids) {
  missing_patients <- setdiff(clinical_donor_ids, rownames(df))
  if (length(missing_patients) > 0) {
    na_frame <- data.frame(matrix(rep(NA, length(missing_patients) * ncol(df)), nrow = length(missing_patients), ncol = ncol(df)))
    colnames(na_frame) <- colnames(df)
    rownames(na_frame) <- missing_patients
    if (!"data.frame" %in% class(df)) {
      df <- data.frame(df, check.names = FALSE)
    }
    df <- rbind(df, na_frame)
  } else if (!"data.frame" %in% class(df)) {
    df <- data.frame(df, check.names = FALSE)
  }
}


prepare_gex_icgc <- function(gex, type = "seq", log = TRUE) {
  if (type == "seq") {
    gex <- gex %>%
      # In case there are multiple normalized read counts for a specific gene
      # where all of donor, specimen, sample id and the intervals are identical,
      # we mean their values.
      reshape2::dcast(icgc_donor_id + icgc_specimen_id + icgc_sample_id + specimen_interval + analyzed_sample_interval ~ gene_id, value.var = "normalized_read_count", fill = 0, fun.aggregate = function(x) mean(x, na.rm = TRUE)) %>%
      data.frame()
  } else {
    gex <- gex %>%
      # In case there are multiple normalized expression values for a specific gene
      # where all of donor, specimen, sample id and the intervals are identical,
      # we mean their values.
      reshape2::dcast(icgc_donor_id + icgc_specimen_id + icgc_sample_id + specimen_interval + analyzed_sample_interval ~ gene_id, value.var = "normalized_expression_value", fill = 0, fun.aggregate = function(x) mean(x, na.rm = TRUE)) %>%
      data.frame()
  }
  gex <- filter_out_duplicates_icgc(gex)
  print(dim(gex))
  gex <- gex %>% dplyr::select(-icgc_specimen_id, -icgc_sample_id, -specimen_interval, -analyzed_sample_interval)
  rownames(gex) <- gex[, 1]
  gex <- gex[, -1]
  if (any(is.na(gex))) {
    gex <- impute_icgc(gex)
  }

  if (log) {
    gex <- log(gex + 1, base = 2)
  }
  return(gex)
}

prepare_icgc <- function(cancer) {
  config <- rjson::fromJSON(
    file = here::here("config", "config.json")
  )
  sample <- vroom::vroom(
    here::here(
      "data", "raw", "ICGC", paste0("sample.", cancer, ".tsv.gz")
    )
  )
  specimen <- vroom::vroom(
    here::here(
      "data", "raw", "ICGC", paste0("specimen.", cancer, ".tsv.gz")
    )
  )
  donor <- vroom::vroom(
    here::here(
      "data", "raw", "ICGC", paste0("donor.", cancer, ".tsv.gz")
    )
  ) %>%
    mutate(
      OS = as.integer(donor_vital_status != "alive")
    ) %>%
    dplyr::rename(
      OS_days = donor_survival_time,
      gender = donor_sex,
      age = donor_age_at_diagnosis,
      tumor_stage = donor_tumour_stage_at_diagnosis,
      cancer_history_relative = cancer_history_first_degree_relative
    ) %>%
    dplyr::select(
      icgc_donor_id,
      OS,
      OS_days,
      gender, age, tumor_stage, cancer_history_relative
    ) %>%
    filter(!is.na(OS) & !is.na(OS_days) & !is.na(age) & !is.na(icgc_donor_id)) %>%
    mutate(cancer_history_relative = recode(cancer_history_relative, `unknown` = "NA"))

  if (config$exposure[[cancer]]) {
    exposure <- vroom::vroom(
      here::here(
        "data", "raw", "ICGC", paste0("donor_exposure.", cancer, ".tsv.gz")
      )
    ) %>%
      mutate(alcohol_history = recode(alcohol_history, `Don't know/Not sure` = "NA")) %>%
      mutate(tobacco_smoking_history_indicator = recode(tobacco_smoking_history_indicator, `Smoking history not documented` = "NA"))

    donor <- donor %>% left_join(y = exposure %>% dplyr::select(icgc_donor_id, tobacco_smoking_history_indicator, alcohol_history))
  }

  donor_admin <- donor[, 1:3, drop = FALSE]
  donor_numerical <- donor[, -(1:3)][, which(sapply(donor[, -(1:3)], function(x) is.numeric(x))), drop = FALSE]
  donor_numerical_dropped_columns <- apply(donor_numerical, 2, function(x) any(is.na(x)))
  donor_numerical <- donor_numerical[, !donor_numerical_dropped_columns]
  donor_categorical <- donor[, -(1:3)][, which(sapply(donor[, -(1:3)], function(x) !is.numeric(x))), drop = FALSE] %>% replace(is.na(.), "NA")
  donor <- cbind(donor_admin, donor_numerical, donor_categorical)
  common_samples <- list(donor$icgc_donor_id)

  if ("gex" %in% config$icgc_modalities[[cancer]]) {
    if (config$gex_type[[cancer]] == "seq") {
      gex <- vroom::vroom(
        here::here(
          "data", "raw", "ICGC", paste0("exp_seq", ".", cancer, ".tsv.gz")
        )
      )
    } else {
      gex <- vroom::vroom(
        here::here(
          "data", "raw", "ICGC", paste0("exp_array", ".", cancer, ".tsv.gz")
        )
      )
    }

    gex <- gex %>%
      mutate(submitted_sample_id = as.character(submitted_sample_id)) %>%
      left_join(sample) %>%
      left_join(specimen, by = c(c("project_code", "icgc_specimen_id", "submitted_specimen_id", "icgc_donor_id", "submitted_donor_id"))) %>%
      filter(grepl("Primary", specimen_type))
    gex <- prepare_gex_icgc(gex, config$gex_type[[cancer]], config$gex_log[[cancer]])
    common_samples <- append(common_samples, list(rownames(gex)))
    gex <- add_missing_modality_samples_icgc(gex, donor$icgc_donor_id)
  }

  if ("mutation" %in% config$icgc_modalities[[cancer]]) {
    mut <- maftools::icgcSimpleMutationToMAF(
      here::here(
        "data", "raw", "ICGC", paste0("simple_somatic_mutation.open", ".", cancer, ".tsv.gz")
      ),
      MAFobj = TRUE
    )
    mut <- data.frame(t(mutCountMatrix(mut, removeNonMutated = FALSE)), check.names = FALSE) %>%
      rownames_to_column("icgc_sample_id") %>%
      left_join(sample) %>%
      left_join(specimen, by = c(c("project_code", "icgc_specimen_id", "submitted_specimen_id", "icgc_donor_id", "submitted_donor_id"))) %>%
      filter(grepl("Primary", specimen_type))
    mut <- prepare_mutation_icgc(mut)
    common_samples <- append(common_samples, list(rownames(mut)))
    mut <- add_missing_modality_samples_icgc(mut, donor$icgc_donor_id)
  }

  common_donors <- Reduce(intersect, common_samples)
  data <- donor %>%
    filter(icgc_donor_id %in% common_donors) %>%
    arrange(desc(icgc_donor_id)) %>%
    rename_with(function(x) paste0("clinical_", x), .cols = -c(OS, OS_days, icgc_donor_id))
  incomplete_data <- donor %>%
    filter(!icgc_donor_id %in% common_donors) %>%
    arrange(desc(icgc_donor_id)) %>%
    rename_with(function(x) paste0("clinical_", x), .cols = -c(OS, OS_days, icgc_donor_id))

  data <- data %>%
    cbind(
      mut %>%
        rownames_to_column() %>%
        filter(rowname %in% common_donors) %>%
        arrange(desc(rowname)) %>%
        dplyr::select(-rowname) %>%
        rename_with(function(x) paste0("mut_", x))
    )

  incomplete_data <- incomplete_data %>%
    cbind(
      mut %>%
        rownames_to_column() %>%
        filter((!rowname %in% common_donors) & rowname %in% common_samples[[1]]) %>%
        arrange(desc(rowname)) %>%
        dplyr::select(-rowname) %>%
        rename_with(function(x) paste0("mut_", x))
    )

  data <- data %>%
    cbind(
      gex %>%
        rownames_to_column() %>%
        filter(rowname %in% common_donors) %>%
        arrange(desc(rowname)) %>%
        dplyr::select(-rowname) %>%
        rename_with(function(x) paste0("gex_", x))
    )

  incomplete_data <- incomplete_data %>%
    cbind(
      gex %>%
        rownames_to_column() %>%
        filter((!rowname %in% common_donors) & rowname %in% common_samples[[1]]) %>%
        arrange(desc(rowname)) %>%
        dplyr::select(-rowname) %>%
        rename_with(function(x) paste0("gex_", x))
    )

  data %>%
    dplyr::rename(patient_id = icgc_donor_id) %>%
    write_csv(
      here::here(
        "data", "processed", "ICGC", paste0(cancer, "_data_complete_modalities_preprocessed.csv")
      )
    )

  incomplete_data %>%
    dplyr::rename(patient_id = icgc_donor_id) %>%
    write_csv(
      here::here(
        "data", "processed", "ICGC", paste0(cancer, "_data_non_complete_modalities_preprocessed.csv")
      )
    )
}
