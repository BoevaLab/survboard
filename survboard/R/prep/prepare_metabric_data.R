prepare_metabric <- function() {
  
  suppressPackageStartupMessages({
    library(maftools)
    library(dplyr)
    library(tidyr)
    library(tibble)
    library(readr)
    
    })
  config <- rjson::fromJSON(
    file = here::here("config", "config.json")
  )
  clinical <- data.frame(vroom::vroom(
    here::here("data_template", "METABRIC", "data_clinical_patient.txt")
  ), check.names = FALSE)
  
  cnv <- data.frame(vroom::vroom(
    here::here("data_template", "METABRIC", "data_cna.txt")
  ), check.names = FALSE) %>% distinct(Hugo_Symbol, .keep_all = TRUE)
  
  gex <- data.frame(vroom::vroom(
    here::here("data_template", "METABRIC", "data_mrna_illumina_microarray.txt")
  ), check.names = FALSE) %>% distinct(Hugo_Symbol, .keep_all = TRUE)
  
  meth <- data.frame(vroom::vroom(
    here::here("data_template", "METABRIC", "data_methylation_promoters_rrbs.txt")
  ), check.names=FALSE) %>% distinct(Hugo_Symbol, .keep_all = TRUE)
  
  mutation <- maftools::read.maf(here::here("data_template", "METABRIC", "data_mutations.txt"))
  # Create patient by gene matrix for mutation. We count only non-silent mutations
  # (which is the default of `maftools::mutCountMatrix`) and keep non mutated
  # genes (for the the pancancer setting).
  mut_master <- mutCountMatrix(mutation,
                               removeNonMutated = FALSE
  )
  
  
  clinical <- clinical[-(1:4), ]
  
  preprocess_clinical_metabric <- function(dataset) {
    # Take out variables of interest
    dataset <- dataset[, unname(sapply(c("#Patient Identifier", "Age at Diagnosis", "Cellularity", "Inferred Menopausal State", "Primary Tumor Laterality", "Tumor Other Histologic Subtype", "Overall Survival (Months)", "Overall Survival Status"), function(x) grep(x, colnames(clinical), fixed = TRUE)))]
    
    # Rename regular columns
    dataset <- dataset %>% 
      rename(patient_id =  `#Patient Identifier`) %>%
      rename(OS_days = `Overall Survival (Months)`) %>%
      rename(OS = `Overall Survival Status`) %>%
      # Take out NAs for OS and OS_days
      filter(!is.na(OS)) %>%
      filter(!is.na(OS_days)) %>%
      # Transform months to days
      dplyr::mutate(OS_days = 30.44 * as.numeric(OS_days)) %>%
      # Transform OS to binary
      dplyr::mutate(OS = ifelse(OS == "0:LIVING", 0, 1)) %>%
      # Rename misc variables
      rename(age = `Age at Diagnosis`) %>%
      rename(cellularity = `Cellularity`) %>%
      rename(menopausal_state = `Inferred Menopausal State`) %>%
      rename(laterality = `Primary Tumor Laterality`) %>%
      rename(histology = `Tumor Other Histologic Subtype`)  %>%
      # Remove patients that were not at risk at beginning of the study
      filter(OS_days != 0) %>%
      # Replace missings with a separate category
      # Filter menopausal state because it is only a single sample
      # that is missing
      mutate(cellularity = replace_na(cellularity, ".MISSING")) %>%
      filter(!is.na(menopausal_state)) %>%
      mutate(laterality = replace_na(laterality, ".MISSING")) %>%
      mutate(histology = replace_na(histology, ".MISSING"))
    
    return(dataset)
    
  }
  
  impute <- function(df) {
    # Columns contain patients - we exclude patients which are missing for more
    # than 10% of all patients.
    large_missing <- which(apply(df, 1, function(x) sum(is.na(x))) > round(dim(df)[2] / 10))
    if (length(large_missing) > 0) {
      df <- df[-which(apply(df, 1, function(x) sum(is.na(x))) > round(dim(df)[2] / 10)), ]
    }
    
    # Any missing values still left over after this initial filtering step are
    # imputed using the median value per feature.
    if (any(is.na(df))) {
      for (na_count in which(apply(df, 1, function(x) any(is.na(x))))) {
        df[na_count, ] <- as.numeric(df[na_count, ]) %>% replace_na(median(as.numeric(df[na_count, ]), na.rm = TRUE))
      }
    }
    return(df)
  }
  
  
  preprocess <- function(df, log = FALSE) {
    if (any(is.na(df))) {
      df <- impute(df)
    }
    if (log) {
      df <- log(1 + df, base = 2)
    }
    return(df)
  }
  
  #' Performs complete preprocessing of METABRIC CNV data.
  #'
  #' @param cnv data.frame. data.frame containing CNV data to be preprocessed.
  #' @param keep_non_primary_samples logical. Whether metastatic and recurrent samples
  #'                                          should also be kept.
  #' @returns data.frame. Preprocessed data.frame.
  prepare_cnv <- function(cnv) {
    rownames(cnv) <- cnv[, 1]
    cnv <- cnv[, 2:ncol(cnv)]
    cnv <- preprocess(cnv, log = FALSE)
    return(cnv)
  }
  
  prepare_gex <- function(gex) {
    rownames(gex) <- gex[, 1]
    gex <- gex[, 2:ncol(gex)]
    gex <- preprocess(gex, log = TRUE)
    return(gex)
  }
  
  prepare_mutation <- function(mut) {
    mut <- preprocess(mut, log = FALSE)
    mut
  }
  
  prepare_meth <- function(meth) {
    rownames(meth) <- meth[, 1]
    meth <- meth[, 2:ncol(meth)]
    meth <- preprocess(meth, log = FALSE)
    return(meth)
  }
  
  
  
  clinical <- preprocess_clinical_metabric(clinical)
  
  cnv <- prepare_cnv(cnv[, -2])
  
  
  gex <- prepare_gex(gex[, -2])
  mut <- prepare_mutation(mut_master)
  meth <- prepare_meth(meth[, -2])
  
  
  meth[1:3, 1:3]
  
  # CNV is complete
  # GEX is complete
  
  # Mut is incomplete
  
  # Meth is incomplete
  
  joint_patients <- Reduce(intersect, list(clinical$patient_id, colnames(cnv), colnames(gex), colnames(mut), colnames(meth)))
  missing_patients <- setdiff(clinical$patient_id, joint_patients)
  
  joint_df <- clinical %>% filter(patient_id %in% joint_patients) %>%
    arrange(desc(patient_id)) %>%
    rename_with(function(x) paste0("clinical_", x), .cols = -c(OS, OS_days, patient_id)) %>%
    cbind(
      data.frame(t(cnv), check.names = FALSE) %>% rownames_to_column %>% filter(rowname %in% joint_patients) %>% arrange(desc(rowname)) %>% select(-rowname) %>%
        rename_with(function(x) paste0("cnv_", x))
    ) %>%
    cbind(
      data.frame(t(gex), check.names = FALSE) %>% rownames_to_column %>% filter(rowname %in% joint_patients) %>% arrange(desc(rowname)) %>% select(-rowname) %>%
        rename_with(function(x) paste0("gex_", x))
    ) %>%
    cbind(
      data.frame(t(mut), check.names = FALSE) %>% rownames_to_column %>% filter(rowname %in% joint_patients) %>% arrange(desc(rowname)) %>% select(-rowname) %>%
        rename_with(function(x) paste0("mut_", x))
    ) %>%
    cbind(
      data.frame(t(meth), check.names = FALSE) %>% rownames_to_column %>% filter(rowname %in% joint_patients) %>% arrange(desc(rowname)) %>% select(-rowname) %>%
        rename_with(function(x) paste0("meth_", x))
    )
  
  missing_df <- clinical %>% dplyr::filter(patient_id %in% missing_patients) %>%
    arrange(desc(patient_id)) %>%
    rename_with(function(x) paste0("clinical_", x), .cols = -c(OS, OS_days, patient_id)) %>%
    cbind(
      data.frame(t(cnv), check.names = FALSE) %>% rownames_to_column %>% filter(rowname %in% missing_patients) %>% arrange(desc(rowname)) %>% select(-rowname) %>%
        rename_with(function(x) paste0("cnv_", x))
    ) %>%
    cbind(
      data.frame(t(gex), check.names = FALSE) %>% rownames_to_column %>% filter(rowname %in% missing_patients) %>% arrange(desc(rowname)) %>% select(-rowname) %>%
        rename_with(function(x) paste0("gex_", x))
    )
  
  missing_df_mut <- data.frame(t(mut), check.names = FALSE) %>% rownames_to_column %>% filter(rowname %in% missing_patients) %>% arrange(desc(rowname)) 
  missing_df_meth <- data.frame(t(meth), check.names = FALSE) %>% rownames_to_column %>% filter(rowname %in% missing_patients) %>% arrange(desc(rowname)) 
  missing_df_mut_append <- matrix(rep(NA, ncol(missing_df_mut) * (length(missing_patients) - nrow(missing_df_mut))), nrow = (length(missing_patients) - nrow(missing_df_mut)))
  missing_df_meth_append <- matrix(rep(NA, ncol(missing_df_meth) * (length(missing_patients) - nrow(missing_df_meth))), nrow = (length(missing_patients) - nrow(missing_df_meth)))
  
  
  missing_df_mut_append[, 1] <- setdiff(missing_patients, missing_df_mut$rowname)
  missing_df_meth_append[, 1] <- setdiff(missing_patients, missing_df_meth$rowname)
  colnames(missing_df_mut_append) <- colnames(missing_df_mut)
  missing_df_mut <- rbind(missing_df_mut, missing_df_mut_append)
  
  colnames(missing_df_meth_append) <- colnames(missing_df_meth)
  missing_df_meth <- rbind(missing_df_meth, missing_df_meth_append)
  
  
  missing_df <- missing_df %>% cbind(missing_df_mut %>% select(-rowname) %>% rename_with(function(x) paste0("mut_", x))) %>% cbind(missing_df_meth %>% select(-rowname) %>% rename_with(function(x) paste0("meth_", x)))
  
  joint_df %>%
    write_csv(
      here::here(
        "data_reproduced", "METABRIC", paste0("BRCA", "_data_complete_modalities_preprocessed.csv")
      )
    )
  
  missing_df %>%
    write_csv(
      here::here(
        "data_reproduced", "METABRIC", paste0("BRCA", "_data_incomplete_modalities_preprocessed.csv")
      )
    )
  return(NULL)
}
