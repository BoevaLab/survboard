prepare_target <- function() {
    suppressPackageStartupMessages({
    library(maftools)
    library(dplyr)
    library(tidyr)
    library(tibble)
    library(readr)
    
  })
  config <- rjson::fromJSON(
    file = here::here(
      "config", "config.json")
  )
  for (cancer in config$target_cancers) {
    cancer_modalities <- config$target_modalities[[cancer]]
    
    # GEX and clinical are always present
    clinical <- data.frame(vroom::vroom(
      here::here(
        "data_template", "TARGET", cancer, "data_clinical_patient.txt")
    ), check.names = FALSE)
    
    sample <- data.frame(vroom::vroom(
      here::here(
        "data_template", "TARGET", cancer, "data_clinical_sample.txt")
    ), check.names = FALSE)
    sample <- sample[-(1:4), ]
    
    gex <- data.frame(vroom::vroom(
      here::here(
        "data_template", "TARGET", cancer, "data_mrna_seq_rpkm.txt")
    ), check.names = FALSE) %>% distinct(Hugo_Symbol, .keep_all = TRUE) %>%
      filter(!is.na(Hugo_Symbol))
    
    if ("mut" %in% cancer_modalities) {
      mutation <- maftools::read.maf(here::here(
        "data_template", "TARGET", cancer, "data_mutations.txt"))
      # Create patient by gene matrix for mutation. We count only non-silent mutations
      # (which is the default of `maftools::mutCountMatrix`) and keep non mutated
      # genes (for the the pancancer setting).
      mut_master <- mutCountMatrix(mutation,
                                   removeNonMutated = FALSE
      )
    }
    
    if ("mirna" %in% cancer_modalities) {
      mirna <- data.frame(vroom::vroom(
        here::here(
          "data_template", "TARGET", cancer, "data_mirna.txt")
      ), check.names=FALSE) %>% distinct(Hugo_Symbol, .keep_all = TRUE) %>%
        filter(!is.na(Hugo_Symbol))
    }
    
    if ("meth" %in% cancer_modalities) {
      meth <- data.frame(vroom::vroom(
        here::here(
          "data_template", "TARGET", cancer, "data_methylation_hm450.txt")
      ), check.names=FALSE) %>% distinct(Hugo_Symbol, .keep_all = TRUE) %>%
        filter(!is.na(Hugo_Symbol))
      
    }
    
    if ("cnv" %in% cancer_modalities) {
      cnv <- data.frame(vroom::vroom(
        here::here(
          "data_template", "TARGET", cancer, "data_cna.txt")
      ), check.names=FALSE) %>% distinct(Hugo_Symbol, .keep_all = TRUE) %>%
        filter(!is.na(Hugo_Symbol))
    }
    
    preprocess_clinical_metabric <- function(dataset) {
      
      # WT -Neoplasm American Joint Committee on Cancer Clinical Group Stage -> Stage
      # NBL - INSS Stage and Tumor Sample Histology
      # ALL -  None
      if (cancer == "ALL") {
        dataset <- dataset[, unname(sapply(c("#Patient Identifier", "Diagnosis Age (days)", 
                                             "Sex", 
                                             "Ethnicity Category",
                                             "Race Category",
                                             "Overall Survival Days", "Overall Survival Status"), function(x) grep(x, colnames(clinical), fixed = TRUE)))]
        
        dataset <- dataset %>% 
          rename(patient_id =  `#Patient Identifier`) %>%
          rename(OS_days = `Overall Survival Days`) %>%
          rename(OS = `Overall Survival Status`) %>%
          # Take out NAs for OS and OS_days
          filter(!is.na(OS)) %>%
          filter(!is.na(OS_days)) %>%
          # Transform months to days
          dplyr::mutate(OS_days = as.numeric(OS_days)) %>%
          
          # Transform OS to binary
          dplyr::mutate(OS = ifelse(OS == "0:LIVING", 0, 1)) %>%
          # Rename misc variables
          rename(age = `Diagnosis Age (days)`) %>%
          dplyr::mutate(age = as.numeric(age) / 365.25) %>%
          rename(gender = `Sex`) %>%
          rename(ethnicity = `Ethnicity Category`) %>%
          rename(race = `Race Category`) %>%
          # Remove patients that were not at risk at beginning of the study
          filter(OS_days > 0) 
      }
      
      else if (cancer == "NBL") {
        dataset <- dataset[, unname(sapply(c("#Patient Identifier", "Diagnosis Age (days)", 
                                             "INSS Stage", "Tumor Sample Histology",
                                             "Sex", 
                                             "Ethnicity Category",
                                             "Race Category",
                                             "Overall Survival Days", "Overall Survival Status"), function(x) grep(x, colnames(clinical), fixed = TRUE)))]
        dataset <- dataset %>% 
          rename(patient_id =  `#Patient Identifier`) %>%
          rename(OS_days = `Overall Survival Days`) %>%
          rename(OS = `Overall Survival Status`) %>%
          # Take out NAs for OS and OS_days
          filter(!is.na(OS)) %>%
          filter(!is.na(OS_days)) %>%
          # Transform months to days
          dplyr::mutate(OS_days = as.numeric(OS_days)) %>%
          
          # Transform OS to binary
          dplyr::mutate(OS = ifelse(OS == "0:LIVING", 0, 1)) %>%
          # Rename misc variables
          rename(age = `Diagnosis Age (days)`) %>%
          dplyr::mutate(age = as.numeric(age) / 365.25) %>%
          rename(gender = `Sex`) %>%
          rename(ethnicity = `Ethnicity Category`) %>%
          rename(race = `Race Category`) %>%
          rename(stage = `INSS Stage`) %>%
          rename(histology = `Tumor Sample Histology`) %>%
          # Remove patients that were not at risk at beginning of the study
          filter(OS_days > 0) 
      }
      else if (cancer == "WT") {
        dataset <- dataset[, unname(sapply(c("#Patient Identifier", "Diagnosis Age (days)", 
                                             "Neoplasm American Joint Committee on Cancer Clinical Group Stage",
                                             "Sex", 
                                             "Ethnicity Category",
                                             "Race Category",
                                             "Overall Survival Days", "Overall Survival Status"), function(x) grep(x, colnames(clinical), fixed = TRUE)))]
        dataset <- dataset %>% 
          rename(patient_id =  `#Patient Identifier`) %>%
          rename(OS_days = `Overall Survival Days`) %>%
          rename(OS = `Overall Survival Status`) %>%
          # Take out NAs for OS and OS_days
          filter(!is.na(OS)) %>%
          filter(!is.na(OS_days)) %>%
          # Transform months to days
          dplyr::mutate(OS_days = as.numeric(OS_days)) %>%
          
          # Transform OS to binary
          dplyr::mutate(OS = ifelse(OS == "0:LIVING", 0, 1)) %>%
          # Rename misc variables
          rename(age = `Diagnosis Age (days)`) %>%
          dplyr::mutate(age = as.numeric(age) / 365.25) %>%
          rename(gender = `Sex`) %>%
          rename(ethnicity = `Ethnicity Category`) %>%
          # Take out missing gender since filling it in doesn't make a lot of sense
          filter(!is.na(gender)) %>%
          rename(race = `Race Category`) %>%
          rename(stage = `Neoplasm American Joint Committee on Cancer Clinical Group Stage`) %>%
          # Remove patients that were not at risk at beginning of the study
          filter(OS_days > 0) 
      }
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
    
    prepare_cnv <- function(cnv) {
      rownames(cnv) <- cnv[, 1]
      cnv <- cnv[, 2:ncol(cnv)]
      # Select only VALIDATION samples
      cnv <- cnv[, unname(sapply(unique(substr(colnames(cnv), 1, nchar(colnames(cnv)) - 3)), function(x) grep(x, colnames(cnv))[1]))]
      cnv <- preprocess(cnv, log = FALSE)
      colnames(cnv) <- substr(colnames(cnv), 1, nchar(colnames(cnv)) -3)
      return(cnv)
    }
    
    prepare_gex <- function(gex) {
      rownames(gex) <- gex[, 1]
      gex <- gex[, 2:ncol(gex)]
      # Select only VALIDATION samples
      gex <- gex[, unname(sapply(unique(substr(colnames(gex), 1, nchar(colnames(gex)) - 3)), function(x) grep(x, colnames(gex))[1]))]
      gex <- preprocess(gex, log = TRUE)
      colnames(gex) <- substr(colnames(gex), 1, nchar(colnames(gex)) -3)
      return(gex)
    }
    
    prepare_mutation <- function(mut) {
      mut <- mut[, unname(sapply(unique(substr(colnames(mut), 1, nchar(colnames(mut)) - 3)), function(x) grep(x, colnames(mut))[1]))]
      mut <- preprocess(mut, log = FALSE)
      colnames(mut) <- substr(colnames(mut), 1, nchar(colnames(mut)) -3)
      mut
    }
    
    prepare_mirna <- function(mirna) {
      rownames(mirna) <- mirna[, 1]
      mirna <- mirna[, 2:ncol(mirna)]
      # Select only VALIDATION samples
      mirna <- mirna[, unname(sapply(unique(substr(colnames(mirna), 1, nchar(colnames(mirna)) - 3)), function(x) grep(x, colnames(mirna))[1]))]
      mirna <- preprocess(mirna, log = TRUE)
      colnames(mirna) <- substr(colnames(mirna), 1, nchar(colnames(mirna)) -3)
      return(mirna)
    }
    
    prepare_meth <- function(meth) {
      rownames(meth) <- meth[, 1]
      meth <- meth[, 2:ncol(meth)]
      # Select only VALIDATION samples
      meth <- meth[, unname(sapply(unique(substr(colnames(meth), 1, nchar(colnames(meth)) - 3)), function(x) grep(x, colnames(meth))[1]))]
      meth <- preprocess(meth, log = FALSE)
      colnames(meth) <- substr(colnames(meth), 1, nchar(colnames(meth)) -3)
      return(meth)
    }
    
    
    clinical <- clinical[-(1:4), ]
    clinical <- preprocess_clinical_metabric(clinical)
    
    gex <- prepare_gex(gex[, -2])
    
    if ("mut" %in% cancer_modalities) {
      mut <- prepare_mutation(mut_master)
    }
    
    if ("mirna" %in% cancer_modalities) {
      mirna <- prepare_mirna(mirna[, -2])
    }
    
    if ("cnv" %in% cancer_modalities) {
      cnv <- prepare_cnv(cnv[, -2])
    }
    
    if ("meth" %in% cancer_modalities) {
      meth <- prepare_meth(meth[, -2])
    }
    
    if (cancer == "WT") {
      joint_patients <- Reduce(intersect, list(clinical$patient_id, colnames(gex), colnames(mirna), colnames(cnv), colnames(meth)))
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
          data.frame(t(mirna), check.names = FALSE) %>% rownames_to_column %>% filter(rowname %in% joint_patients) %>% arrange(desc(rowname)) %>% select(-rowname) %>%
            rename_with(function(x) paste0("mirna_", x))
        ) %>%
        cbind(
          data.frame(t(meth), check.names = FALSE) %>% rownames_to_column %>% filter(rowname %in% joint_patients) %>% arrange(desc(rowname)) %>% select(-rowname) %>%
            rename_with(function(x) paste0("meth_", x))
        )
      
      missing_df <- clinical %>% dplyr::filter(patient_id %in% missing_patients) %>%
        arrange(desc(patient_id)) %>%
        rename_with(function(x) paste0("clinical_", x), .cols = -c(OS, OS_days, patient_id))
      
      missing_df_gex <- data.frame(t(gex), check.names = FALSE) %>% rownames_to_column %>% filter(rowname %in% missing_patients) %>% arrange(desc(rowname)) 
      missing_df_mirna <- data.frame(t(mirna), check.names = FALSE) %>% rownames_to_column %>% filter(rowname %in% missing_patients) %>% arrange(desc(rowname))
      missing_df_cnv <- data.frame(t(cnv), check.names = FALSE) %>% rownames_to_column %>% filter(rowname %in% missing_patients) %>% arrange(desc(rowname))
      missing_df_meth <- data.frame(t(meth), check.names = FALSE) %>% rownames_to_column %>% filter(rowname %in% missing_patients) %>% arrange(desc(rowname)) 
      
      
      missing_df_gex_append <- matrix(rep(NA, ncol(missing_df_gex) * (length(missing_patients) - nrow(missing_df_gex))), nrow = (length(missing_patients) - nrow(missing_df_gex)))
      
      missing_df_mirna_append <- matrix(rep(NA, ncol(missing_df_mirna) * (length(missing_patients) - nrow(missing_df_mirna))), nrow = (length(missing_patients) - nrow(missing_df_mirna)))
      
      missing_df_cnv_append <- matrix(rep(NA, ncol(missing_df_cnv) * (length(missing_patients) - nrow(missing_df_cnv))), nrow = (length(missing_patients) - nrow(missing_df_cnv)))
      
      missing_df_meth_append <- matrix(rep(NA, ncol(missing_df_meth) * (length(missing_patients) - nrow(missing_df_meth))), nrow = (length(missing_patients) - nrow(missing_df_meth)))
      
      
      missing_df_gex_append[, 1] <- setdiff(missing_patients, missing_df_gex$rowname)
      missing_df_mirna_append[, 1] <- setdiff(missing_patients, missing_df_mirna$rowname)
      missing_df_cnv_append[, 1] <- setdiff(missing_patients, missing_df_cnv$rowname)
      missing_df_meth_append[, 1] <- setdiff(missing_patients, missing_df_meth$rowname)
      
      colnames(missing_df_gex_append) <- colnames(missing_df_gex)
      missing_df_gex <- rbind(missing_df_gex, missing_df_gex_append)
      
      colnames(missing_df_mirna_append) <- colnames(missing_df_mirna)
      missing_df_mirna <- rbind(missing_df_mirna, missing_df_mirna_append)
      
      colnames(missing_df_cnv_append) <- colnames(missing_df_cnv)
      missing_df_cnv <- rbind(missing_df_cnv, missing_df_cnv_append)
      
      colnames(missing_df_meth_append) <- colnames(missing_df_meth)
      missing_df_meth <- rbind(missing_df_meth, missing_df_meth_append)
      
      missing_df <- missing_df %>% cbind(missing_df_cnv  %>% select(-rowname) %>% rename_with(function(x) paste0("cnv_", x))) %>% cbind(missing_df_meth %>% select(-rowname) %>% rename_with(function(x) paste0("meth_", x))) %>%
        cbind(missing_df_mirna %>% select(-rowname) %>% rename_with(function(x) paste0("mirna_", x))) %>%
        cbind(missing_df_gex %>% select(-rowname) %>% rename_with(function(x) paste0("gex_", x)))
      
    } else if (cancer == "ALL") {
      joint_patients <- Reduce(intersect, list(clinical$patient_id, colnames(cnv), colnames(gex), colnames(mirna)))
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
          data.frame(t(mirna), check.names = FALSE) %>% rownames_to_column %>% filter(rowname %in% joint_patients) %>% arrange(desc(rowname)) %>% select(-rowname) %>%
            rename_with(function(x) paste0("mirna_", x))
        )
      
      missing_df <- clinical %>% dplyr::filter(patient_id %in% missing_patients) %>%
        arrange(desc(patient_id)) %>%
        rename_with(function(x) paste0("clinical_", x), .cols = -c(OS, OS_days, patient_id))
      
      missing_df_gex <- data.frame(t(gex), check.names = FALSE) %>% rownames_to_column %>% filter(rowname %in% missing_patients) %>% arrange(desc(rowname)) 
      missing_df_mirna <- data.frame(t(mirna), check.names = FALSE) %>% rownames_to_column %>% filter(rowname %in% missing_patients) %>% arrange(desc(rowname))
      missing_df_cnv <- data.frame(t(cnv), check.names = FALSE) %>% rownames_to_column %>% filter(rowname %in% missing_patients) %>% arrange(desc(rowname))
      
      missing_df_gex_append <- matrix(rep(NA, ncol(missing_df_gex) * (length(missing_patients) - nrow(missing_df_gex))), nrow = (length(missing_patients) - nrow(missing_df_gex)))
      
      missing_df_mirna_append <- matrix(rep(NA, ncol(missing_df_mirna) * (length(missing_patients) - nrow(missing_df_mirna))), nrow = (length(missing_patients) - nrow(missing_df_mirna)))
      
      missing_df_cnv_append <- matrix(rep(NA, ncol(missing_df_cnv) * (length(missing_patients) - nrow(missing_df_cnv))), nrow = (length(missing_patients) - nrow(missing_df_cnv)))
      
      missing_df_gex_append[, 1] <- setdiff(missing_patients, missing_df_gex$rowname)
      missing_df_mirna_append[, 1] <- setdiff(missing_patients, missing_df_mirna$rowname)
      missing_df_cnv_append[, 1] <- setdiff(missing_patients, missing_df_cnv$rowname)
      
      colnames(missing_df_gex_append) <- colnames(missing_df_gex)
      missing_df_gex <- rbind(missing_df_gex, missing_df_gex_append)
      
      colnames(missing_df_mirna_append) <- colnames(missing_df_mirna)
      missing_df_mirna <- rbind(missing_df_mirna, missing_df_mirna_append)
      
      colnames(missing_df_cnv_append) <- colnames(missing_df_cnv)
      missing_df_cnv <- rbind(missing_df_cnv, missing_df_cnv_append)
      
      
      missing_df <- missing_df %>% cbind(missing_df_cnv  %>% select(-rowname) %>% rename_with(function(x) paste0("cnv_", x))) %>%
        cbind(missing_df_mirna %>% select(-rowname) %>% rename_with(function(x) paste0("mirna_", x))) %>%
        cbind(missing_df_gex %>% select(-rowname) %>% rename_with(function(x) paste0("gex_", x)))
      
    }
    
    
    joint_df %>%
      write_csv(
        here::here(
          "data_reproduced", "TARGET", paste0(cancer, "_data_complete_modalities_preprocessed.csv")
        )
      )
    
    missing_df %>%
      write_csv(
        here::here(
          "data_reproduced", "TARGET", paste0(cancer, "_data_incomplete_modalities_preprocessed.csv")
        )
      )
  }


}
