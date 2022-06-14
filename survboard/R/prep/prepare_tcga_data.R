library(dplyr)
library(readr)
library(rjson)
library(fastDummies)
library(tidyr)
library(janitor)
library(forcats)

choose_proper_sample <- function(barcodes) {
  vials <- substr(barcodes, 16, 16)
  if (length(unique(vials)) > 1) {
    return(str_sort(barcodes, numeric = TRUE)[1])
  } else {
    return(str_sort(barcodes, numeric = TRUE, decreasing = TRUE)[1])
  }
}

filter_samples <- function(df) {
  types <- substr(colnames(df), 14, 15)
  # Select only primary tumors
  # 01 - Primary Solid Tumor
  # 03 - Primary Blood Derived Cancer - Peripheral Blood
  # 09 - Primary Blood Derived Cancer - Bone Marrow
  df <- df[, types %in% c("01", "03", "09")]
  donors <- substr(colnames(df), 1, 12)
  duplicated_donors <- names(which(table(donors) > 1))
  if (length(duplicated_donors) > 0) {
    unique_df <- df[, !donors %in% duplicated_donors]
    df <- cbind(unique_df, df[, unname(sapply(
      unlist(lapply(duplicated_donors, function(x) choose_proper_sample(grep(x, colnames(df), value = TRUE)))),
      function(x) grep(x, colnames(df))
    ))])
  }
  colnames(df) <- unname(sapply(colnames(df), function(x) substr(x, 1, 12)))
  return(df)
}
filter_out_duplicates <- function(df) {
  duplicated <- which(table(sapply(colnames(df), function(x) substr(x, 1, 12))) > 1)
  df_filtered <- data.frame(df[, -unlist(sapply(names(duplicated), function(x) grep(x, colnames(df))))], check.names = FALSE)
  df_filtered[, unname(sapply(names(duplicated), function(x) sort(grep(x, colnames(df), value = TRUE))[1]))] <- df[, unname(sapply(unname(sapply(names(duplicated), function(x) sort(grep(x, colnames(df), value = TRUE))[1])), function(x) grep(x, colnames(df))))]
  return(df_filtered)
}
impute <- function(df) {
  large_missing <- which(apply(df, 1, function(x) sum(is.na(x))) > round(dim(df)[2] / 10))
  if (length(large_missing) > 0) {
    df <- df[-which(apply(df, 1, function(x) sum(is.na(x))) > round(dim(df)[2] / 10)), ]
  }

  if (any(is.na(df))) {
    for (na_count in which(apply(df, 1, function(x) any(is.na(x))))) {
      df[na_count, ] <- as.numeric(df[na_count, ]) %>% replace_na(median(as.numeric(df[na_count, ]), na.rm = TRUE))
    }
  }
  return(df)
}

preprocess <- function(df, log = FALSE) {
  if (all(nchar(colnames(df)) >= 15)) {
    df <- filter_samples(df)
  }
  if (any(is.na(df))) {
    df <- impute(df)
  }
  if (log) {
    df <- log(1 + df, base = 2)
  }
  return(df)
}

prepare_gene_expression_pancan <- function(gex) {
  rownames(gex) <- gex[, 1]
  gex <- gex[, 2:ncol(gex)]
  gex <- preprocess(gex, log = TRUE)
  return(gex)
}

prepare_clinical_data <- function(clinical_raw, clinical_ext_raw, cancer, standard = TRUE) {
  clinical <- clinical_raw %>%
    filter(type == cancer) %>%
    # remove any patients for which the OS endpoint is missing
    filter(!(is.na(OS) | is.na(OS.time))) %>%
    # remove any patients which were not at risk at the start of the study
    filter(!(OS.time == 0)) %>%
    dplyr::select(
      bcr_patient_barcode, OS, OS.time,
      age_at_initial_pathologic_diagnosis,
      gender,
      race,
      ajcc_pathologic_tumor_stage,
      clinical_stage,
      histological_type
    ) %>%
    mutate(race = recode(race, `[Unknown]` = "NA", `[Not Available]` = "NA", `[Not Evaluated]` = "NA")) %>%
    mutate(ajcc_pathologic_tumor_stage = recode(ajcc_pathologic_tumor_stage, `[Unknown]` = "NA", `[Not Available]` = "NA", `[Discrepancy]` = "NA", `[Not Applicable]` = "NA")) %>%
    mutate(clinical_stage = recode(clinical_stage, `[Not Available]` = "NA", `[Discrepancy]` = "NA", `[Not Applicable]` = "NA")) %>%
    mutate(histological_type = recode(histological_type, `[Unknown]` = "NA", `[Not Available]` = "NA", `[Discrepancy]` = "NA", `[Not Applicable]` = "NA")) %>%
    mutate(race = replace_na(race, "NA")) %>%
    mutate(ajcc_pathologic_tumor_stage = replace_na(ajcc_pathologic_tumor_stage, "NA")) %>%
    mutate(clinical_stage = replace_na(clinical_stage, "NA")) %>%
    mutate(histological_type = replace_na(histological_type, "NA"))
  admin <- clinical[, c("bcr_patient_barcode", "OS", "OS.time")]
  colnames(admin)[1] <- "patient_id"
  numerical <- clinical[, names(which(sapply(clinical[, -(1:3)], is.numeric))), drop = FALSE]
  na_numerical_mask <- apply(numerical, 2, function(x) any(is.na(x)))
  categorical <- clinical[, names(which(!sapply(clinical[, -(1:3)], is.numeric)))]
  clinical <- cbind(admin, numerical[, !na_numerical_mask], categorical)
  return(clinical)
}

prepare_cnv <- function(cnv) {
  rownames(cnv) <- cnv[, 1]
  cnv <- cnv[, 2:ncol(cnv)]
  cnv <- preprocess(cnv, log = FALSE)
  cnv
}

prepare_meth_pancan <- function(meth) {
  rownames(meth) <- meth[, 1]
  meth <- meth[, 2:ncol(meth)]
  meth <- preprocess(meth, log = FALSE)
  meth
}

prepare_mutation <- function(mut) {
  mut <- preprocess(mut, log = FALSE)
  mut
}

prepare_rppa_pancan <- function(rppa, duplicates = "keep") {
  rownames(rppa) <- rppa[, 1]
  rppa <- t(rppa[, 2:ncol(rppa)])
  rppa <- preprocess(rppa, log = FALSE)
  rppa
}

prepare_mirna_pancan <- function(mirna, duplicates = "keep", logbase = 2, offset = 2) {
  rownames(mirna) <- mirna[, 1]
  mirna <- mirna[, 2:ncol(mirna)]
  mrina <- preprocess(mirna, log = TRUE)
}

append_missing_modality_samples <- function(df, barcodes) {
  if (!all(barcodes %in% colnames(df))) {
    if (!"data.frame" %in% class(df)) {
      df <- data.frame(df, check.names = FALSE)
    }
    df[, setdiff(barcodes, colnames(df))] <- NA
  }
  return(df)
}

prepare_new_cancer_dataset <- function(cancer, include_rppa = FALSE, include_mirna = TRUE, include_mutation = TRUE, include_methylation = TRUE, standard_clinical = TRUE, include_gex = TRUE, include_cnv = TRUE) {
  config <- rjson::fromJSON(
    file = here::here("config", "config.json")
  )
  tcga_cdr_master <- tcga_cdr
  tcga_w_followup_master <- tcga_w_followup
  clinical <- prepare_clinical_data(tcga_cdr_master, tcga_w_followup_master, cancer = cancer, standard = TRUE)
  sample_barcodes <- list(clinical$patient_id)
  if (include_gex) {
    patients <- unname(unlist(sapply(clinical$patient_id, function(x) grep(x, colnames(gex_master)))))
    gex_filtered <- gex_master[, c(1, patients)]
    gex <- prepare_gene_expression_pancan(gex_filtered)
    sample_barcodes <- append(sample_barcodes, list(colnames(gex)))
    gex <- append_missing_modality_samples(gex, clinical$patient_id)
  }

  if (include_cnv) {
    patients <- unname(unlist(sapply(clinical$patient_id, function(x) grep(x, colnames(cnv_master)))))
    cnv_filtered <- cnv_master[, c(1, patients)]
    cnv <- prepare_cnv(cnv_filtered)
    sample_barcodes <- append(sample_barcodes, list(colnames(cnv)))
    cnv <- append_missing_modality_samples(cnv, clinical$patient_id)
  }
  if (include_methylation) {
    patients <- unname(unlist(sapply(clinical$patient_id, function(x) grep(x, colnames(meth_master)))))
    meth_filtered <- meth_master[, c(1, patients)]
    meth <- prepare_meth_pancan(data.frame(meth_filtered, check.names = FALSE))
    sample_barcodes <- append(sample_barcodes, list(colnames(meth)))
    meth <- append_missing_modality_samples(meth, clinical$patient_id)
  }
  if (include_rppa) {
    patients <- unlist(sapply(clinical$patient_id, function(x) grep(x, rppa_master$SampleID)))
    if (length(patients) > 0) {
      rppa_filtered <- rppa_master[patients, -c(2)]
      rppa <- prepare_rppa_pancan(data.frame(rppa_filtered, check.names = FALSE))
      sample_barcodes <- append(sample_barcodes, list(colnames(rppa)))
      rppa <- append_missing_modality_samples(rppa, clinical$patient_id)
    } else {
      include_rppa <- FALSE
    }
  }

  if (include_mirna) {
    patients <- unname(unlist(sapply(clinical$patient_id, function(x) grep(x, colnames(mirna_master)))))
    if (length(patients) > 0) {
      mirna_filtered <- mirna_master[, c(1, patients)]
      mirna <- prepare_mirna_pancan(data.frame(mirna_filtered, check.names = FALSE))
      sample_barcodes <- append(sample_barcodes, list(colnames(mirna)))
      mirna <- append_missing_modality_samples(mirna, clinical$patient_id)
    } else {
      include_mirna <- FALSE
    }
  }
  if (include_mutation) {
    patients <- unname(unlist(sapply(clinical$patient_id, function(x) grep(x, colnames(mut_master)))))
    mut_filtered <- mut_master[, patients]
    mutation <- prepare_mutation(mut_filtered)
    sample_barcodes <- append(sample_barcodes, list(colnames(mutation)))
    mutation <- append_missing_modality_samples(mutation, clinical$patient_id)
  }
  common_samples <- Reduce(intersect, sample_barcodes)
  data <- clinical %>%
    filter(patient_id %in% common_samples) %>%
    arrange(desc(patient_id)) %>%
    rename_with(function(x) paste0("clinical_", x), .cols = -c(OS, OS.time, patient_id))
  if (include_cnv) {
    data <- data %>%
      cbind(
        data.frame(t(cnv), check.names = FALSE) %>%
          rownames_to_column() %>%
          filter(rowname %in% common_samples) %>%
          arrange(desc(rowname)) %>%
          dplyr::select(-rowname) %>%
          rename_with(function(x) paste0("cnv_", x))
      )
  }

  if (include_gex) {
    data <- data %>%
      cbind(
        data.frame(t(gex), check.names = FALSE) %>%
          rownames_to_column() %>%
          filter(rowname %in% common_samples) %>%
          arrange(desc(rowname)) %>%
          dplyr::select(-rowname) %>%
          rename_with(function(x) paste0("gex_", x))
      )
  }


  if (include_methylation) {
    data <- data %>%
      cbind(
        data.frame(t(meth), check.names = FALSE) %>%
          rownames_to_column() %>%
          filter(rowname %in% common_samples) %>%
          arrange(desc(rowname)) %>%
          dplyr::select(-rowname) %>%
          rename_with(function(x) paste0("meth_", x))
      )
  }

  if (include_mirna) {
    data <- data %>%
      cbind(
        data.frame(t(mirna), check.names = FALSE) %>%
          rownames_to_column() %>%
          filter(rowname %in% common_samples) %>%
          arrange(desc(rowname)) %>%
          dplyr::select(-rowname) %>%
          rename_with(function(x) paste0("mirna_", x))
      )
  }

  if (include_mutation) {
    data <- data %>%
      cbind(
        data.frame(t(mutation), check.names = FALSE) %>%
          rownames_to_column() %>%
          filter(rowname %in% common_samples) %>%
          arrange(desc(rowname)) %>%
          dplyr::select(-rowname) %>%
          rename_with(function(x) paste0("mutation_", x))
      )
  }

  if (include_rppa) {
    data <- data %>%
      cbind(
        data.frame(t(rppa), check.names = FALSE) %>%
          rownames_to_column() %>%
          filter(rowname %in% common_samples) %>%
          arrange(desc(rowname)) %>%
          dplyr::select(-rowname) %>%
          rename_with(function(x) paste0("rppa_", x))
      )
  }
  data %>%
    rename(OS_days = OS.time) %>%
    write_csv(
      here::here("data", "processed", "TCGA", paste0(cancer, "_data_complete_modalities_preprocessed.csv"))
    )

  data <- clinical %>%
    filter(!patient_id %in% common_samples) %>%
    arrange(desc(patient_id)) %>%
    rename_with(function(x) paste0("clinical_", x), .cols = -c(OS, OS.time, patient_id))

  if (include_cnv) {
    data <- data %>%
      cbind(
        data.frame(t(cnv), check.names = FALSE) %>%
          rownames_to_column() %>%
          filter((!(rowname %in% common_samples)) & rowname %in% sample_barcodes[[1]]) %>%
          arrange(desc(rowname)) %>%
          dplyr::select(-rowname) %>%
          rename_with(function(x) paste0("cnv_", x))
      )
  }

  if (include_gex) {
    data <- data %>%
      cbind(
        data.frame(t(gex), check.names = FALSE) %>%
          rownames_to_column() %>%
          filter((!(rowname %in% common_samples)) & rowname %in% sample_barcodes[[1]]) %>%
          arrange(desc(rowname)) %>%
          dplyr::select(-rowname) %>%
          rename_with(function(x) paste0("gex_", x))
      )
  }


  if (include_methylation) {
    data <- data %>%
      cbind(
        data.frame(t(meth), check.names = FALSE) %>%
          rownames_to_column() %>%
          filter((!(rowname %in% common_samples)) & rowname %in% sample_barcodes[[1]]) %>%
          arrange(desc(rowname)) %>%
          dplyr::select(-rowname) %>%
          rename_with(function(x) paste0("meth_", x))
      )
  }

  if (include_mirna) {
    data <- data %>%
      cbind(
        data.frame(t(mirna), check.names = FALSE) %>%
          rownames_to_column() %>%
          filter((!(rowname %in% common_samples)) & rowname %in% sample_barcodes[[1]]) %>%
          arrange(desc(rowname)) %>%
          dplyr::select(-rowname) %>%
          rename_with(function(x) paste0("mirna_", x))
      )
  }

  if (include_mutation) {
    data <- data %>%
      cbind(
        data.frame(t(mutation), check.names = FALSE) %>%
          rownames_to_column() %>%
          filter((!(rowname %in% common_samples)) & rowname %in% sample_barcodes[[1]]) %>%
          arrange(desc(rowname)) %>%
          dplyr::select(-rowname) %>%
          rename_with(function(x) paste0("mut_", x))
      )
  }

  if (include_rppa) {
    data <- data %>%
      cbind(
        data.frame(t(rppa), check.names = FALSE) %>%
          rownames_to_column() %>%
          filter((!(rowname %in% common_samples)) & rowname %in% sample_barcodes[[1]]) %>%
          arrange(desc(rowname)) %>%
          dplyr::select(-rowname) %>%
          rename_with(function(x) paste0("rppa_", x))
      )
  }
  data %>%
    rename(OS_days = OS.time) %>%
    write_csv(
      here::here(
        "data", "processed", "TCGA", paste0(cancer, "_data_non_complete_modalities_preprocessed.csv")
      )
    )
}
