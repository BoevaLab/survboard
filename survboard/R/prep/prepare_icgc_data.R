suppressPackageStartupMessages({
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
})

#' Impute data. We use the same imputation logic as for TCGA (see `impute` function in `prepare_tcga_data.R`).
#'
#' @param df data.frame. data.frame containing some missing values that are to be imputed.
#'                       Notably, in ICGC, patients are in the ROWS, not the columns
#'                       as in TCGA.
#'
#' @returns data.frame. Complete data.frame for which all missing values have been
#'                      either imputed or the covariates in question have been removed.
impute_icgc <- function(df) {
  # Rows contain patients - we exclude patients which are missing for more
  # than 10% of all patients.
  big_missing_genes <- which(apply(df, 2, function(x) sum(is.na(x))) > round(dim(df)[1] / 10))
  if (length(big_missing_genes) > 0) {
    df <- df[, -big_missing_genes]
  }

  # Any missing values still left over after this initial filtering step are
  # imputed using the median value per feature.
  median_per_gene <- apply(df, 2, function(x) median(x, na.rm = TRUE))
  median_per_gene <- split(unname(median_per_gene), names(median_per_gene))
  df <- df %>% replace_na(median_per_gene)
  return(df)
}

#' Filter out duplicate samples for ICGC. We note that for ICGC, some patients
#' had multiple samples taken, often at different times. There is also a distinction
#' between specimens (tissue taken from a patient) and samples (part of a specimen)
#' to be analysed at a particular time (see https://docs.icgc.org/submission/guide/clinical-data-submission-file-specifications/)
#' for details. We note that since our goal is to use molecular data as close
#' as possible to diagnosis, we always selected the specimen
#' and sample with the minimum time (i.e., that was closest to diagnosis).
#' Sometimes this information is not available, in which we randomly selected
#' a sample.
#'
#' @param df data.frame. data.frame containing information on specimens and samples.
#'
#' @returns data.frame. Filtered data.frame which contains a unique combination
#'                      of specimen and sample for every patient.
filter_out_duplicates_icgc <- function(df) {
  # We take a copy of `df` as our main data.frame to work with.
  main_frame <- df
  # Calculate how many samples and specimens we have for every patient.
  indicator_frame <- df %>%
    group_by(icgc_donor_id) %>%
    summarise(
      n_samples = n_distinct(icgc_sample_id),
      n_specimens = n_distinct(icgc_specimen_id)
    )

  # We first deal with donors which have more than one specimen.
  duplicated_specimen_donors <- indicator_frame %>%
    filter(n_specimens > 1) %>%
    pull(icgc_donor_id)

  # We remove any specimens which are either (i) not equal to the minimum
  # specimen interval (i.e., that aren't maximally close to the time of diagnosis)
  # or (ii) are NA. Since this may remove all specimens for a particular
  # donor (e.g., if all specimen_intervals are NA), we add one of them
  # back in at random later on.
  specimens_to_remove <- main_frame %>%
    filter(icgc_donor_id %in% duplicated_specimen_donors) %>%
    group_by(icgc_donor_id) %>%
    filter(specimen_interval != min(specimen_interval, na.rm = TRUE) | is.na(specimen_interval)) %>%
    pull(icgc_specimen_id)

  df <- df %>% filter(!icgc_specimen_id %in% specimens_to_remove)

  # Since now all donors have a maximum of one specimen, we deal with multiple
  # samples for this specimen next.
  duplicated_sample_donors <- indicator_frame %>%
    filter(n_samples > 1 & n_specimens == 1) %>%
    pull(icgc_donor_id)

  # We apply to the same logic that we did on the specimen level earlier to
  # the sample level.
  samples_to_remove <- main_frame %>%
    filter(icgc_donor_id %in% duplicated_sample_donors) %>%
    group_by(icgc_donor_id) %>%
    filter(analyzed_sample_interval != min(analyzed_sample_interval, na.rm = TRUE) | is.na(analyzed_sample_interval)) %>%
    pull(icgc_sample_id)

  df <- df %>% filter(!icgc_sample_id %in% samples_to_remove)

  # If any patients were in the original df but are now no longer in `df`,
  # they were removed because all of their specimen or sample types were NA.
  # In these cases, we randomly sample one of the patients and add them back.
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
  # Lastly, we randomly sample one specimen and sample combiantion for any
  # patients which still have more than one sample or specimen. This may happen
  # e.g., if more than one specimen or sample were taken at the same time
  # (and thus multiple specimens/samples) are equal to the minimum time.
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


#' Performs complete preprocessing of ICGC mutation data.
#'
#' @param mut data.frame. data.frame containing mutation data to be preprocessed.
#'
#' @returns data.frame. Preprocessed data.frame.
prepare_mutation_icgc <- function(mut) {
  # Remove duplicates.
  mut <- filter_out_duplicates_icgc(mut)
  # Remove specimen_type and so on - only keep the donor ID and mutation
  # data (which starts with ENSG for all genes).
  mut <- mut %>% dplyr::select(icgc_donor_id, starts_with("ENSG"))
  # Switch donor_id to the rownames.
  rownames(mut) <- mut[, 1]
  mut <- mut[, -1]
  # Impute data, if necessary.
  if (any(is.na(mut))) {
    mut <- impute_icgc(mut)
  }
  return(mut)
}

#' Appends missing modality samples to complete modality samples for a specific modality, such
#' that they can easily be separated later. Simply adds all NA columns for the missing modality samples.
#' Also see `append_missing_modality_samples` in `prepare_tcga_data.R`.
#'
#' @param df data.frame. data.frame containing complete samples for a specific modality.
#' @param clinical_donor_ids character. Vector of barcodes from the clinical data (since clinical)
#'                            data is always present.
#'
#' @returns data.frame.
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

#' Performs complete preprocessing of ICGC mRNA data.
#'
#' @param mut data.frame. data.frame containing mRNA data to be preprocessed.
#'
#' @returns data.frame. Preprocessed data.frame.
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

#' Helper function to perform complete preprocessing for ICGC datasets. Writes
#' datasets directly to disk, separated by complete and missing modality samples.
#' General logic is identical to `prepare_new_cancer_dataset` in `prepare_tcga_data.R`.
#'
#' @param cancer character. Cancer dataset to be prepared.
#' @param keep_non_primary_samples logical. Whether the recurrent and metastatic samples
#'                                          should be included in the dataset.
#' @param keep_patients_without_survival_information logical. Whether patients with
#'                                                            missing survival information
#'                                                            should be included in the dataset.
#'
#' @returns NULL.
prepare_icgc <- function(cancer, keep_non_primary_samples = FALSE, keep_patients_without_survival_information = FALSE) {
  config <- rjson::fromJSON(
    file = here::here("config", "config.json")
  )
  sample <- vroom::vroom(
    here::here(
      "data_template", "ICGC", paste0("sample.", cancer, ".tsv.gz")
    )
  )
  specimen <- vroom::vroom(
    here::here(
      "data_template", "ICGC", paste0("specimen.", cancer, ".tsv.gz")
    )
  )
  donor <- vroom::vroom(
    here::here(
      "data_template", "ICGC", paste0("donor.", cancer, ".tsv.gz")
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
    mutate(tumor_stage = paste0("stage", as.character(tumor_stage))) %>%
    dplyr::select(
      icgc_donor_id,
      OS,
      OS_days,
      gender, age, tumor_stage, cancer_history_relative
    )
  if (!keep_patients_without_survival_information) {
    donor <- donor %>% filter(!is.na(OS) & !is.na(OS_days))
  }
  donor <- donor %>%
    filter(!is.na(age) & !is.na(icgc_donor_id)) %>%
    mutate(cancer_history_relative = recode(cancer_history_relative, `unknown` = ".MISSING"))

  if (config$exposure[[cancer]]) {
    exposure <- vroom::vroom(
      here::here(
        "data_template", "ICGC", paste0("donor_exposure.", cancer, ".tsv.gz")
      )
    ) %>%
      mutate(alcohol_history = recode(alcohol_history, `Don't know/Not sure` = ".MISSING")) %>%
      mutate(tobacco_smoking_history_indicator = recode(tobacco_smoking_history_indicator, `Smoking history not documented` = ".MISSING"))

    donor <- donor %>% left_join(y = exposure %>% dplyr::select(icgc_donor_id, tobacco_smoking_history_indicator, alcohol_history))
  }

  donor_admin <- donor[, 1:3, drop = FALSE]
  donor_numerical <- donor[, -(1:3)][, which(sapply(donor[, -(1:3)], function(x) is.numeric(x))), drop = FALSE]
  donor_numerical_dropped_columns <- apply(donor_numerical, 2, function(x) any(is.na(x)))
  donor_numerical <- donor_numerical[, !donor_numerical_dropped_columns]
  donor_categorical <- donor[, -(1:3)][, which(sapply(donor[, -(1:3)], function(x) !is.numeric(x))), drop = FALSE] %>% replace(is.na(.), ".MISSING")
  donor <- cbind(donor_admin, donor_numerical, donor_categorical)
  common_samples <- list(donor$icgc_donor_id)

  if ("gex" %in% config$icgc_modalities[[cancer]]) {
    if (config$gex_type[[cancer]] == "seq") {
      gex <- vroom::vroom(
        here::here(
          "data_template", "ICGC", paste0("exp_seq", ".", cancer, ".tsv.gz")
        )
      )
    } else {
      gex <- vroom::vroom(
        here::here(
          "data_template", "ICGC", paste0("exp_array", ".", cancer, ".tsv.gz")
        )
      )
    }

    gex <- gex %>%
      mutate(submitted_sample_id = as.character(submitted_sample_id)) %>%
      left_join(sample) %>%
      left_join(specimen, by = c(c("project_code", "icgc_specimen_id", "submitted_specimen_id", "icgc_donor_id", "submitted_donor_id")))
    if (keep_non_primary_samples) {
      gex <- gex %>% filter(grepl("(Primary|Recurrent)", specimen_type))
    } else {
      gex <- gex %>% filter(grepl("Primary", specimen_type))
    }

    gex <- prepare_gex_icgc(gex, config$gex_type[[cancer]], config$gex_log[[cancer]])
    common_samples <- append(common_samples, list(rownames(gex)))
    gex <- add_missing_modality_samples_icgc(gex, donor$icgc_donor_id)
  }

  if ("mut" %in% config$icgc_modalities[[cancer]]) {
    mut <- maftools::icgcSimpleMutationToMAF(
      here::here(
        "data_template", "ICGC", paste0("simple_somatic_mutation.open", ".", cancer, ".tsv.gz")
      ),
      MAFobj = TRUE
    )
    mut <- data.frame(t(mutCountMatrix(mut, removeNonMutated = FALSE)), check.names = FALSE) %>%
      rownames_to_column("icgc_sample_id") %>%
      left_join(sample) %>%
      left_join(specimen, by = c(c("project_code", "icgc_specimen_id", "submitted_specimen_id", "icgc_donor_id", "submitted_donor_id")))
    if (keep_non_primary_samples) {
      mut <- mut %>% filter(grepl("(Primary|Recurrent)", specimen_type))
    } else {
      mut <- mut %>% filter(grepl("Primary", specimen_type))
    }
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
    # Rename to `OS_days` for consistency with other projects/datasets.
    dplyr::rename(patient_id = icgc_donor_id) %>%
    write_csv(
      here::here(
        "data_reproduced", "ICGC", paste0(cancer, "_data_complete_modalities_preprocessed.csv")
      )
    )

  incomplete_data %>%
    # Rename to `OS_days` for consistency with other projects/datasets.
    dplyr::rename(patient_id = icgc_donor_id) %>%
    write_csv(
      here::here(
        "data_reproduced", "ICGC", paste0(cancer, "_data_incomplete_modalities_preprocessed.csv")
      )
    )
  return(NULL)
}
